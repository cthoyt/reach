"""
./bioresources/target/classes/application.conf:229:            labels = [CellType]
./bioresources/src/main/resources/application.conf:229:            labels = [CellType]
./main/target/scala-2.12/classes/org/clulab/reach/biogrammar/entities/entities.yml:184:  #   label: [CellType]
./main/src/main/resources/org/clulab/reach/biogrammar/entities/entities.yml:184:  #   label: [CellType]
"""

import json
import os

import pyobo
from kestrel.sources.literature.utils import get_pubmed_dataframe
from kestrel.ner.vaccine_grounder import get_vaccine_terms, get_vaccine_grounder
from indra.literature import pubmed_client
from bioregistry import curie_to_str
from pathlib import Path
import click
import pandas as pd
from tqdm import tqdm
import gilda.ner

HERE = Path(__file__).parent.resolve()
BIORESOURCES_DIRECTORY = HERE.joinpath(
    "bioresources",
    "src",
    "main",
    "resources",
    "org",
    "clulab",
    "reach",
    "kb",
)
BIORESOURCES_PATH = BIORESOURCES_DIRECTORY.joinpath("vaccines.tsv")

REACH_DIRECTORY = Path.home().joinpath("Documents", "reach")
REACH_PAPERS_DIRECTORY = REACH_DIRECTORY.joinpath("papers")
REACH_OUTPUT_DIRECTORY = REACH_DIRECTORY.joinpath("output")
REACH_TSV_PATH = REACH_DIRECTORY.joinpath("reach_results.tsv")
GILDA_TSV_PATH = REACH_DIRECTORY.joinpath("gilda_results.tsv")
SUMMARY_PATH = REACH_DIRECTORY.joinpath("missing_summary.tsv")

JAVA_HOME = Path("/Library/Java/JavaVirtualMachines/zulu-11.jdk/Contents/Home")

#: This corresponds to the label in entities.yml
VACCINE_TYPE = "vaccine"


def main():
    # create_resources_file()
    # get_papers()
    # run_cli()
    process_output()


def create_resources_file():
    if not BIORESOURCES_PATH.is_file():
        click.echo(f"Creating Reach resource at {BIORESOURCES_PATH}")
        terms = get_vaccine_terms()
        with BIORESOURCES_PATH.open("w") as file:
            for term in tqdm(terms, unit="term", desc="Writing Reach resource"):
                print(term.text, curie_to_str(term.db, term.id), sep="\t", file=file)


def get_papers():
    pmids = pubmed_client.get_ids("vaccine")
    df = get_pubmed_dataframe(pmids)
    for pmid, data in tqdm(
        df.iterrows(), unit="paper", desc="Writing article TSVs", unit_scale=True
    ):
        REACH_PAPERS_DIRECTORY.joinpath(f"{pmid}.txt").write_text(data["abstract"])


def run_cli():
    args = [
        "sbt",
        "-java-home",
        JAVA_HOME.as_posix(),
        '"runMain org.clulab.reach.RunReachCLI"',
    ]
    command = " ".join(args)
    click.echo(f"Running: {command}")
    os.system(command)


def process_output():
    grounder = get_vaccine_grounder()

    reach_rows = []
    for path in tqdm(
        list(REACH_OUTPUT_DIRECTORY.glob("*.uaz.entities.json")),
        unit="article",
        unit_scale=True,
        desc="REACH Grounding",
    ):
        stem = int(path.stem.split(".")[0])
        results = json.loads(path.read_text())
        for entity in results["frames"]:
            if entity.get("type") != "vaccine":
                continue

            text = entity["text"]
            start = entity["start-pos"]["offset"]
            end = entity["end-pos"]["offset"]

            scored_matches = grounder.ground(text)
            if not scored_matches:
                curie = None
                name = None
                score = None
            else:
                scored_match = scored_matches[0]
                curie = curie_to_str(scored_match.term.db, scored_match.term.id)
                name = pyobo.get_name_by_curie(curie)
                score = round(scored_match.score, 2)
            reach_rows.append((stem, start, end, text, curie, name, score))

    columns = ["pubmed", "start", "end", "text", "curie", "name", "score"]
    reach_df = pd.DataFrame(
        sorted(reach_rows),
        columns=columns,
    )
    click.echo(f"Writing {len(reach_rows):,} Reach results to {REACH_TSV_PATH}")
    reach_df.to_csv(REACH_TSV_PATH, sep="\t", index=False)

    reach_df[reach_df["curie"].isna()].to_csv(SUMMARY_PATH, sep="\t", index=False)

    gilda_rows = []
    for path in tqdm(
        list(REACH_PAPERS_DIRECTORY.glob("*.txt")),
        unit="article",
        unit_scale=True,
        desc="Gilda NER",
    ):
        text = path.read_text()
        pubmed = path.stem
        annotations = gilda.ner.annotate(text, grounder=grounder)
        for span, scored_match, start, stop in annotations:
            curie = curie_to_str(scored_match.term.db, scored_match.term.id)
            gilda_rows.append(
                (
                    pubmed,
                    start,
                    stop,
                    span,
                    curie,
                    pyobo.get_name_by_curie(curie),
                    round(scored_match.score, 2),
                )
            )
    gilda_df = pd.DataFrame(sorted(gilda_rows), columns=columns)
    click.echo(f"Writing {len(gilda_rows):,} Gilda results to {GILDA_TSV_PATH}")
    gilda_df.to_csv(GILDA_TSV_PATH, sep="\t", index=False)


if __name__ == "__main__":
    main()
