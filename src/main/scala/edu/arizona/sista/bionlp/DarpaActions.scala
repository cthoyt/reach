package edu.arizona.sista.bionlp

import edu.arizona.sista.processors.Document
import edu.arizona.sista.struct.Interval
import edu.arizona.sista.odin._
import edu.arizona.sista.bionlp.mentions._

class DarpaActions extends Actions {

  def splitSimpleEvents(mentions: Seq[Mention], state: State): Seq[Mention] = mentions flatMap {
    case m: EventMention if m matches "SimpleEvent" =>
      if (m.arguments.keySet contains "cause") {
        val cause = m.arguments("cause")
        val evArgs = m.arguments - "cause"
        val ev = new BioEventMention(
          m.labels, m.trigger, evArgs, m.sentence, m.document, m.keep, m.foundBy)
        val regArgs = Map("controlled" -> Seq(ev), "controller" -> cause)
        val reg = new BioRelationMention(
          Seq("Positive_regulation", "ComplexEvent", "Event"),
          regArgs, m.sentence, m.document, m.keep, m.foundBy)
        Seq(reg, ev)
      } else Seq(m.toBioMention)
    case m => Seq(m.toBioMention)
  }

  // FIXME this is an ugly hack that has to go
  override val identity: Action = splitSimpleEvents

  /** This action handles the creation of mentions from labels generated by the NER system.
    * Rules that use this action should run in an iteration following and rules recognizing
    * "custom" entities. This action will only create mentions if no other mentions overlap
    * with a NER label sequence.
    */
  def mkNERMentions(mentions: Seq[Mention], state: State): Seq[Mention] = {
    mentions flatMap { m =>
      val candidates = state.mentionsFor(m.sentence, m.tokenInterval.toSeq)
      // do any candidates intersect the mention?
      val overlap = candidates.exists(_.tokenInterval.overlaps(m.tokenInterval))
      if (overlap) None else Some(m)
    }
  }

  /** This action handles the creation of ubiquitination EventMentions.
    * A Ubiquitination event cannot involve arguments (theme/cause) with the text Ubiquitin.
    */
  def mkUbiquitination(mentions: Seq[Mention], state: State): Seq[Mention] = {
    val filteredMentions = mentions.filter { m =>
      // Don't allow Ubiquitin
      !m.arguments.values.flatten.exists(_.text.toLowerCase.startsWith("ubiq")) 
    }
    filteredMentions.map(_.toBioMention)
  }

  /** This action handles the creation of Binding EventMentions for rules using token patterns.
    * Currently Odin does not support the use of arguments of the same name in Token patterns.
    * Because of this, we have adopted the convention of following duplicate names with a
    * unique number (ex. theme1, theme2).
    * mkBinding simply unifies named arguments of this type (ex. theme1 & theme2 -> theme)
    */
  def mkBinding(mentions: Seq[Mention], state: State): Seq[Mention] = mentions flatMap {
    case m: EventMention =>
      val arguments = m.arguments
      val themes = for {
        name <- arguments.keys.toSeq
        if name startsWith "theme"
        theme <- arguments(name)
      } yield theme
      // remove bindings with less than two themes
      if (themes.size < 2) Nil
      // if binding has two themes we are done
      else if (themes.size == 2) {
        val args = Map("theme" -> themes)
        Seq(new BioEventMention(
          m.labels, m.trigger, args, m.sentence, m.document, m.keep, m.foundBy))
      } else {
        // binarize bindings
        // return bindings with pairs of themes
        for (pair <- themes.combinations(2)) yield {
          val args = Map("theme" -> pair)
          new BioEventMention(m.labels, m.trigger, args, m.sentence, m.document, m.keep, m.foundBy)
        }
      }
  }

}
