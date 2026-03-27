# Graph CLI

Inspect and edit the knowledge graph from the command line.

## Commands

### `acervo graph show`

List all nodes in the graph.

```bash
acervo graph show                    # table: ID, Label, Type, Kind, Layer, Facts, Edges
acervo graph show --kind entity      # filter by kind (entity, file, symbol, section)
acervo graph show --json             # JSON output for piping
```

### `acervo graph show <id>`

Show full detail for a single node.

```bash
acervo graph show batman
```

Output includes: facts, edges, linked files, attributes, chunk_ids.

### `acervo graph search <query>`

Search nodes by label and fact content.

```bash
acervo graph search "React"
acervo graph search "React" --kind entity
acervo graph search "React" --json
```

### `acervo graph delete <id>`

Delete a node and all its edges. Prompts for confirmation.

```bash
acervo graph delete batman           # interactive confirmation
acervo graph delete batman --yes     # skip confirmation
```

### `acervo graph merge <keep_id> <absorb_id>`

Merge two nodes. The first node is kept, the second is absorbed (facts merged, edges transferred, second node deleted).

```bash
acervo graph merge batman the_batman        # interactive preview + confirmation
acervo graph merge batman the_batman --yes  # skip confirmation
```

### `acervo graph repair`

Detect and fix graph corruption. Checks for:

- Nodes missing required fields (id, label, type, kind)
- Edges referencing non-existent nodes
- Duplicate edges
- Invalid chunk_ids

```bash
acervo graph repair
# Output: "Graph is healthy" or repair summary
```

## REST API

All graph operations are also available via REST when the proxy is running:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/acervo/graph/nodes` | GET | List all nodes |
| `/acervo/graph/nodes/{id}` | GET | Node detail |
| `/acervo/graph/search?q=...` | GET | Search nodes |
| `/acervo/graph/nodes/{id}` | DELETE | Delete node |
| `/acervo/graph/merge` | POST | Merge two nodes |
