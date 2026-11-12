"""LESSON 06a - MCP: package your tools once, use them from any agent.

MCP (Model Context Protocol) is a standard way to expose tools/data to AI apps.
Instead of hard-wiring tools into one script (lessons 01-05), you run a small SERVER;
any MCP CLIENT (Claude Desktop, Claude Code, your own agent in 06b) can discover and
call them. It's like writing a REST API, but for models.

Run standalone (it waits for a client on stdin/stdout - that's normal):
    python lessons/06_mcp_server.py

Use it from Claude Desktop: Settings -> Developer -> Edit Config, add:
    {"mcpServers": {"credit-unions": {
        "command": "/ABSOLUTE/PATH/TO/python",
        "args": ["/ABSOLUTE/PATH/TO/agent-lab/lessons/06_mcp_server.py"]}}}
Restart Claude Desktop and ask: "Using the credit-unions tools, which CU had the highest
delinquency in 2024?"

Use it from Claude Code:
    claude mcp add credit-unions -- python /ABSOLUTE/PATH/lessons/06_mcp_server.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentlab import data_tools as dt  # noqa: E402

try:                                    # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

server = Server("credit-unions")


# With MCP, the schema is inferred from type hints + docstring. Same tools as before,
# so these are thin typed wrappers around agentlab.data_tools.

@server.tool()
def describe_dataset() -> dict:
    """Columns, row count, date range and summary stats of the credit-union panel. Call first."""
    return dt.describe_dataset.fn()


@server.tool()
def query_data(columns: list[str], where: str = "", group_by: str = "", agg: str = "mean") -> dict:
    """Filter/aggregate the panel. `where` is a pandas query string, e.g. "quarter >= '2024Q1'".
    agg is one of mean|sum|median|max|min. Returns at most 60 rows."""
    return dt.query_data.fn(columns, where, group_by, agg)


@server.tool()
def run_regression(y: str, x: list[str], where: str = "", entity_fe: bool = False) -> dict:
    """OLS of y on x with SEs clustered by credit union; entity_fe adds CU fixed effects."""
    return dt.run_regression.fn(y, x, where, entity_fe)


@server.resource("panel://schema")
def schema() -> str:
    """Resources are read-only context the client can attach (vs tools, which the model calls)."""
    return "cu_id, quarter, loans_bn, delinquency_rate(%), unemployment(%), policy_rate(%), hpi_yoy(%), mortgage_share"


if __name__ == "__main__":
    server.run()  # stdio transport by default
