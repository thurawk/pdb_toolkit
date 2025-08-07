# pdb_toolkit.py

import os
import requests
import tempfile
from typing import Dict
from Bio.PDB import PDBParser
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("pdb_toolkit")

# Register tools
@mcp.tool()
def fetch_pdb_metadata(pdb_id: str) -> Dict:
    """
    Fetch basic metadata from RCSB PDB for a given pdb_id.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)

    if not response.ok:
        raise ValueError(f"Failed to fetch metadata for PDB ID: {pdb_id}")

    data = response.json()
    return {
        "id": pdb_id.upper(),
        "title": data.get("struct", {}).get("title", "N/A"),
        "deposition_date": data.get("rcsb_accession_info", {}).get("initial_deposition_date", "N/A"),
        "polymer_count": data.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0),
        "experimental_method": data.get("exptl", [{}])[0].get("method", "N/A")
    }

@mcp.tool()
def download_pdb_file(pdb_id: str, format: str = "pdb") -> str:
    """
    Download the structure file (.pdb or .cif) and return the temporary file path.
    """
    format = format.lower()
    if format not in ["pdb", "cif"]:
        raise ValueError("Format must be 'pdb' or 'cif'")

    ext = "pdb" if format == "pdb" else "cif"
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{ext}"
    response = requests.get(url)

    if not response.ok:
        raise ValueError(f"Failed to download structure file for {pdb_id}")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    tmp_file.write(response.content)
    tmp_file.close()

    return tmp_file.name

mcp.tool()
def list_chains(pdb_file: str) -> list[str]:
    """List chain IDs in the given PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    model = next(structure.get_models())
    return [chain.id for chain in model.get_chains()]

@mcp.tool()
def count_residues(pdb_file: str, chain_id: str) -> int:
    """Count the number of residues in a given chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    model = next(structure.get_models())
    for chain in model.get_chains():
        if chain.id == chain_id:
            return len([res for res in chain.get_residues()])
    raise ValueError(f"Chain {chain_id} not found.")

@mcp.tool()
def get_mock_domain_info(pdb_id: str) -> dict:
    """Return mock domain residue ranges for testing."""
    return {
        "pdb_id": pdb_id,
        "domains": [
            {"name": "DomainA", "start": 10, "end": 50},
            {"name": "DomainB", "start": 100, "end": 150}
        ]
    }

if __name__ == "__main__":
    # Initialize and run the MCP server
    mcp.run(transport="stdio")
