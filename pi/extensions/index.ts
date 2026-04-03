import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { StringEnum } from "@mariozechner/pi-ai";
import { spawn, type ChildProcess } from "node:child_process";
import { resolve } from "node:path";
import { createInterface, type Interface as ReadlineInterface } from "node:readline";
import { existsSync } from "node:fs";

// ---------------------------------------------------------------------------
// Minimal MCP client — speaks JSON-RPC 2.0 over stdio to ontomics serve
// ---------------------------------------------------------------------------

class McpClient {
  private proc: ChildProcess;
  private rl: ReadlineInterface;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();

  private constructor(proc: ChildProcess) {
    this.proc = proc;
    this.rl = createInterface({ input: proc.stdout! });
    this.rl.on("line", (line: string) => this.onLine(line));
    proc.stderr?.on("data", () => {});
  }

  static async start(binaryPath: string, cwd: string): Promise<McpClient> {
    const proc = spawn(binaryPath, ["serve"], {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const client = new McpClient(proc);
    await client.request("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "pi-ontomics", version: "1.0.0" },
    });
    client.notify("notifications/initialized", {});
    return client;
  }

  async callTool(
    name: string,
    args: Record<string, unknown>,
  ): Promise<string> {
    const result = (await this.request("tools/call", {
      name,
      arguments: args,
    })) as { content?: Array<{ text?: string }> };
    const text = result.content?.[0]?.text ?? JSON.stringify(result);
    return text;
  }

  dispose(): void {
    this.proc.kill();
    this.rl.close();
  }

  get alive(): boolean {
    return !this.proc.killed && this.proc.exitCode === null;
  }

  private request(method: string, params: unknown): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.write({ jsonrpc: "2.0", id, method, params });
    });
  }

  private notify(method: string, params: unknown): void {
    this.write({ jsonrpc: "2.0", method, params });
  }

  private write(msg: unknown): void {
    this.proc.stdin!.write(JSON.stringify(msg) + "\n");
  }

  private onLine(line: string): void {
    if (!line.trim()) return;
    try {
      const msg = JSON.parse(line) as {
        id?: number;
        result?: unknown;
        error?: { message: string };
      };
      if (msg.id != null && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id)!;
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      }
    } catch {
      // ignore non-JSON lines (e.g. stderr leaks)
    }
  }
}

// ---------------------------------------------------------------------------
// Find the ontomics binary installed by npm postinstall
// ---------------------------------------------------------------------------

function findBinary(): string {
  // Installed by binary-install.js into <package_root>/node_modules/.bin_real/
  const candidates = [
    resolve(__dirname, "..", "node_modules", ".bin_real", "ontomics"),
    resolve(__dirname, "..", "node_modules", ".bin_real", "ontomics.exe"),
  ];
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  throw new Error(
    `ontomics binary not found. Searched:\n${candidates.join("\n")}\n` +
      "Run 'npm rebuild @ontomics/ontomics' or reinstall the package.",
  );
}

// ---------------------------------------------------------------------------
// Tool definitions — mirrors src/tools.rs tool_definitions()
// ---------------------------------------------------------------------------

interface ToolDef {
  mcpName: string;
  label: string;
  description: string;
  promptSnippet: string;
  parameters: ReturnType<typeof Type.Object>;
}

function toolDefs(): ToolDef[] {
  return [
    {
      mcpName: "query_concept",
      label: "Query Concept",
      description:
        "Semantic concept lookup — returns variants (including abbreviations), " +
        "related concepts, naming conventions, function signatures, and file locations.",
      promptSnippet:
        "ontomics_query_concept: semantic concept lookup with variants and relationships",
      parameters: Type.Object({
        term: Type.String({ description: "Concept term to look up" }),
        max_related: Type.Optional(
          Type.Integer({ description: "Max related concepts (default: 10)" }),
        ),
        max_occurrences: Type.Optional(
          Type.Integer({
            description: "Max occurrence locations (default: 5)",
          }),
        ),
        max_variants: Type.Optional(
          Type.Integer({
            description: "Max variant identifiers (default: 20)",
          }),
        ),
        max_signatures: Type.Optional(
          Type.Integer({
            description: "Max function signatures (default: 5)",
          }),
        ),
      }),
    },
    {
      mcpName: "check_naming",
      label: "Check Naming",
      description:
        "Check if an identifier follows project naming conventions. " +
        "Returns consistent/inconsistent verdict with canonical form.",
      promptSnippet:
        "ontomics_check_naming: validate identifier against project conventions",
      parameters: Type.Object({
        identifier: Type.String({
          description: "Identifier to check (e.g. 'n_dims')",
        }),
      }),
    },
    {
      mcpName: "suggest_name",
      label: "Suggest Name",
      description:
        "Generate project-consistent identifier names from a natural language " +
        "description using the project's actual conventions.",
      promptSnippet:
        "ontomics_suggest_name: generate convention-consistent names from description",
      parameters: Type.Object({
        description: Type.String({
          description: "Natural language description (e.g. 'count of features')",
        }),
      }),
    },
    {
      mcpName: "ontology_diff",
      label: "Ontology Diff",
      description:
        "Compare the domain ontology between git revisions — shows concepts " +
        "added, removed, or changed.",
      promptSnippet:
        "ontomics_ontology_diff: diff domain vocabulary across git revisions",
      parameters: Type.Object({
        since: Type.Optional(
          Type.String({
            description: "Git ref to diff from (default: HEAD~5)",
          }),
        ),
      }),
    },
    {
      mcpName: "list_concepts",
      label: "List Concepts",
      description:
        "List the project's domain vocabulary ranked by importance. " +
        "A semantic overview of what this codebase is about.",
      promptSnippet:
        "ontomics_list_concepts: ranked domain vocabulary overview",
      parameters: Type.Object({
        top_k: Type.Optional(
          Type.Integer({ description: "Max concepts to return" }),
        ),
      }),
    },
    {
      mcpName: "list_conventions",
      label: "List Conventions",
      description:
        "List the project's actual naming conventions detected from code — " +
        "prefix/suffix patterns, conversion patterns, casing rules.",
      promptSnippet:
        "ontomics_list_conventions: detected naming conventions with examples",
      parameters: Type.Object({}),
    },
    {
      mcpName: "describe_symbol",
      label: "Describe Symbol",
      description:
        "Describe a function or class without reading its source — returns " +
        "signature, parameters, callers, callees, semantic role.",
      promptSnippet:
        "ontomics_describe_symbol: function/class info without reading source",
      parameters: Type.Object({
        name: Type.String({
          description: "Function or class name (e.g. 'spatial_transform')",
        }),
      }),
    },
    {
      mcpName: "locate_concept",
      label: "Locate Concept",
      description:
        "Find the best entry points for understanding a concept — ranked " +
        "shortlist of key functions, classes, and files to read.",
      promptSnippet:
        "ontomics_locate_concept: find key entry points for a concept",
      parameters: Type.Object({
        term: Type.String({
          description: "Concept to locate (e.g. 'transform')",
        }),
      }),
    },
    {
      mcpName: "export_domain_pack",
      label: "Export Domain Pack",
      description:
        "Export the project's full domain knowledge as portable YAML — " +
        "abbreviations, conventions, domain terms, concept associations.",
      promptSnippet:
        "ontomics_export_domain_pack: export domain knowledge as YAML",
      parameters: Type.Object({}),
    },
    {
      mcpName: "list_entities",
      label: "List Entities",
      description:
        "Find classes and functions matching a semantic role or concept. " +
        "Returns entities with semantic roles and concept tags.",
      promptSnippet:
        "ontomics_list_entities: find functions/classes by concept or role",
      parameters: Type.Object({
        concept: Type.Optional(
          Type.String({ description: "Filter by concept (e.g. 'loss')" }),
        ),
        role: Type.Optional(
          Type.String({
            description: "Filter by semantic role substring (e.g. 'module')",
          }),
        ),
        kind: Type.Optional(
          StringEnum(["class", "function"] as const, {
            description: "Filter by entity kind",
          }),
        ),
        top_k: Type.Optional(
          Type.Integer({ description: "Max entities to return (default: 20)" }),
        ),
      }),
    },
    {
      mcpName: "vocabulary_health",
      label: "Vocabulary Health",
      description:
        "Measure vocabulary health — convention coverage, consistency ratio, " +
        "cluster cohesion, top inconsistencies.",
      promptSnippet:
        "ontomics_vocabulary_health: code quality metrics for naming consistency",
      parameters: Type.Object({}),
    },
    {
      mcpName: "trace_concept",
      label: "Trace Concept",
      description:
        "Trace how a domain concept flows through the codebase via call " +
        "chains — producers, consumers, and bridge functions.",
      promptSnippet:
        "ontomics_trace_concept: trace concept flow through call chains",
      parameters: Type.Object({
        concept: Type.String({
          description: "Concept to trace (e.g. 'transform')",
        }),
        max_depth: Type.Optional(
          Type.Integer({
            description: "Maximum call chain depth (default: 5)",
          }),
        ),
      }),
    },
  ];
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  let client: McpClient | null = null;
  let binaryPath: string;

  try {
    binaryPath = findBinary();
  } catch {
    // Binary not installed — skip tool registration entirely.
    // This can happen on unsupported platforms.
    return;
  }

  async function getClient(): Promise<McpClient> {
    if (client && client.alive) return client;
    client = await McpClient.start(binaryPath, process.cwd());
    return client;
  }

  // Strip undefined keys so the MCP server receives clean arguments
  function cleanArgs(
    params: Record<string, unknown>,
  ): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) out[k] = v;
    }
    return out;
  }

  for (const def of toolDefs()) {
    pi.registerTool({
      name: `ontomics_${def.mcpName}`,
      label: def.label,
      description: def.description,
      promptSnippet: def.promptSnippet,
      promptGuidelines: [
        "Use ontomics tools BEFORE grep/glob for semantic codebase questions.",
      ],
      parameters: def.parameters,
      async execute(_toolCallId, params, _signal, onUpdate, _ctx) {
        onUpdate?.({
          content: [{ type: "text", text: `Querying ontomics: ${def.mcpName}...` }],
        });
        try {
          const mcp = await getClient();
          const text = await mcp.callTool(def.mcpName, cleanArgs(params));
          return { content: [{ type: "text", text }] };
        } catch (err) {
          throw new Error(
            `ontomics ${def.mcpName} failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      },
    });
  }

  // Clean up the subprocess when pi shuts down
  pi.on("session_start", async () => {
    // Fresh session — ensure stale clients are cleaned up
    if (client && !client.alive) {
      client.dispose();
      client = null;
    }
  });
}
