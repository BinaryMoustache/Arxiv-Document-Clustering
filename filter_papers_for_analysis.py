import json
import argparse
import random
from helpers.utils import timeit
from typing import Generator, Dict, List, Optional


def get_metadata(data_file: str) -> Generator[str, None, None]:
    """
    Generator function that reads the JSON file line by line.

    Args:
        data_file (str): The path to the arXiv metadata JSON file.

    Yields:
        str: Each line from the file as a string.
    """
    with open(data_file, 'r') as f:
        for line in f:
            yield line


@timeit
def filter_papers(data_file: str, year_filter: int, max_docs: Optional[int] = None) -> None:
    """
    Processes the arXiv metadata file, filters papers updated after the specified year and cs category,
    and saves the results to a new JSON file. Optionally limits the number of documents.

    Args:
        data_file (str): The path to the arXiv metadata JSON file.
        year_filter (int): The year to filter papers updated in or after this year.
        max_docs (Optional[int]): The maximum number of documents to include in the output file. 
                                  If None, include all filtered papers.

    Returns:
        None

    Notes:
        - Papers are filtered based on the year they were updated, requiring the year to be greater than
          or equal to `year_filter`.
        - Only papers with titles and abstracts are considered.
        - Papers must belong to the "cs." category to be included.
        - The resulting JSON file is saved with the name `filtered_papers_from_<year_filter>.json` in the 
          `data` directory.
        - If `max_docs` is specified, only a random sample of that many papers is included in the output.
    """
    print("Processing Data ..")
    metadata = get_metadata(data_file)

    titles: List[str] = []
    abstracts: List[str] = []
    years: List[int] = []
    categories: List[str] = []

    for paper in metadata:
        paper_dict: Dict[str, str] = json.loads(paper)
        year_updated: int = int(paper_dict.get('update_date', '')[:4])
        paper_categories: List[str] = paper_dict.get("categories", "").split()
        title: str = paper_dict.get("title", "").strip()
        abstract: str = paper_dict.get("abstract", "").strip()

        if year_updated >= year_filter and title and abstract:
            if any(cat.startswith("cs.") for cat in paper_categories):
                titles.append(title)
                abstracts.append(abstract)
                years.append(year_updated)
                categories.append(paper_dict.get("categories", ""))

    results: List[Dict[str, str]] = [
        {
            "title": title,
            "abstract": abstract,
            "year": year,
            "categories": category
        }
        for title, abstract, year, category in zip(titles, abstracts, years, categories)
    ]

    if max_docs is not None and len(results) > max_docs:
        results = random.sample(results, max_docs)

    output_file: str = f'data/filtered_papers_from_{year_filter}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Filtered results saved to {output_file}")
    print(f"Total papers gathered: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Filter arXiv metadata by year and optionally limit the number of results.')

    parser.add_argument('filepath', type=str,
                        help='Path to the arXiv metadata file')
    parser.add_argument(
        'year', type=int, help='Filter papers updated in or after this year')
    parser.add_argument('--max_docs', type=int, default=None,
                        help='Maximum number of documents to include')

    try:
        args: argparse.Namespace = parser.parse_args()
    except SystemExit as e:
        print("\nExample usage:")
        print(f"python script.py data/arxiv_metadata.json 2024 --max_docs 1000")
        print("This will filter papers updated in or after 2024 and limit the results to 1000 papers.")

        parser.print_help()
        exit(1)
    filter_papers(args.filepath, args.year, args.max_docs)
