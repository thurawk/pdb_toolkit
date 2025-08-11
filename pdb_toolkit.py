# pdb_toolkit.py

import os
import requests
from typing import Dict
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pdb_toolkit")

# Register tools
@mcp.tool()
def fetch_pdb(pdb_id: str) -> str:
    """
    Download the structure file in .cif format for a given PDB ID and return the temporary file path.
    """

    pdb_id = pdb_id.strip().upper()
    save_dir = os.path.join("data", "pdb_files")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{pdb_id}.cif")

    if os.path.exists(file_path):
        print(f"Using internal data: {file_path}")
        return file_path

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded and saved to {file_path}")
    return file_path

@mcp.tool()
def read_pdb(file_path: str) -> str:
    """
    Read a PDB or mmCIF file from the given file path and return its raw content as a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    return content

@mcp.tool()
def resolve_alias_to_uniprot(alias: str) -> str:
    """
    Resolve protein alias or name to UniProt ID.
    """
    # Example using UniProt API:
    url = f"https://rest.uniprot.org/uniprotkb/search?query={alias}&format=json&fields=accession"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    # Parse to get the top UniProt ID
    if data.get("results"):
        return data["results"][0]["primaryAccession"]
    else:
        raise ValueError(f"Could not resolve alias '{alias}' to UniProt ID")

@mcp.tool()
def map_uniprot_to_pdb(uniprot_id: str) -> list[str]:
    """
    Map UniProt ID to list of associated PDB IDs.
    Tries RCSB exact match first, then falls back to UniProt's own cross-references.
    """
    pdb_ids = []

    try:
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": uniprot_id
                }
            },
            "return_type": "entry"
        }
        url = "https://search.rcsb.org/rcsbsearch/v1/query"
        r = requests.post(url, json=query, timeout=10)
        r.raise_for_status()
        results = r.json().get("result_set", [])
        pdb_ids = [res["identifier"] for res in results]
    except Exception:
        pass

    if not pdb_ids:
        try:
            uni_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            r = requests.get(uni_url, timeout=10)
            r.raise_for_status()
            data = r.json()
            for ref in data.get("uniProtKBCrossReferences", []):
                if ref.get("database") == "PDB":
                    pdb_ids.append(ref["id"])
        except Exception:
            pass
    pdb_ids = sorted(set(pdb_ids))
    if not pdb_ids:
        raise ValueError(f"No PDB entries found for UniProt ID '{uniprot_id}'")
    return pdb_ids

if __name__ == "__main__":
    # Initialize and run the MCP server
    mcp.run(transport="stdio")
