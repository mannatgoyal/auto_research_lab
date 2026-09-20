"""Research MCP Server definition using Python MCP SDK."""

import sys
from mcp.server.mcpserver import MCPServer
from mcp_servers.research_server.tools import tool_ping_research


def create_research_server() -> MCPServer:
    """Instantiates and configures the Research MCP Server instance."""
    app = MCPServer("autonomous-ml-research")

    @app.tool(
        name="ping_research",
        description="Health check for the autonomous research server.",
    )
    def ping_research() -> dict:
        return tool_ping_research()

    return app


def main():
    """Entrypoint for running the research MCP server over local stdio transport."""
    server = create_research_server()
    print("Starting Autonomous ML Research Server on stdio...", file=sys.stderr)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
