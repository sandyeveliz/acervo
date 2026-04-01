import { getDatabase } from "../config/database";

export interface Todo {
  id: number;
  user_id: number;
  title: string;
  description: string;
  completed: boolean;
  priority: "low" | "medium" | "high";
  due_date: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateTodoInput {
  title: string;
  description?: string;
  priority?: "low" | "medium" | "high";
  due_date?: string;
}

export interface UpdateTodoInput {
  title?: string;
  description?: string;
  completed?: boolean;
  priority?: "low" | "medium" | "high";
  due_date?: string | null;
}

export interface TodoFilters {
  completed?: boolean;
  priority?: "low" | "medium" | "high";
  search?: string;
}

export function findAllByUser(userId: number, filters: TodoFilters = {}): Todo[] {
  const db = getDatabase();
  let query = "SELECT * FROM todos WHERE user_id = ?";
  const params: unknown[] = [userId];

  if (filters.completed !== undefined) {
    query += " AND completed = ?";
    params.push(filters.completed ? 1 : 0);
  }
  if (filters.priority) {
    query += " AND priority = ?";
    params.push(filters.priority);
  }
  if (filters.search) {
    query += " AND (title LIKE ? OR description LIKE ?)";
    params.push(`%${filters.search}%`, `%${filters.search}%`);
  }

  query += " ORDER BY created_at DESC";
  const rows = db.prepare(query).all(...params) as Todo[];
  return rows.map(normalizeTodo);
}

export function findById(id: number, userId: number): Todo | null {
  const db = getDatabase();
  const row = db.prepare("SELECT * FROM todos WHERE id = ? AND user_id = ?").get(id, userId);
  return row ? normalizeTodo(row as Todo) : null;
}

export function create(userId: number, input: CreateTodoInput): Todo {
  const db = getDatabase();
  const result = db.prepare(`
    INSERT INTO todos (user_id, title, description, priority, due_date)
    VALUES (?, ?, ?, ?, ?)
  `).run(userId, input.title, input.description || "", input.priority || "medium", input.due_date || null);

  return findById(result.lastInsertRowid as number, userId)!;
}

export function update(id: number, userId: number, input: UpdateTodoInput): Todo | null {
  const existing = findById(id, userId);
  if (!existing) return null;

  const db = getDatabase();
  const fields: string[] = [];
  const values: unknown[] = [];

  if (input.title !== undefined) { fields.push("title = ?"); values.push(input.title); }
  if (input.description !== undefined) { fields.push("description = ?"); values.push(input.description); }
  if (input.completed !== undefined) { fields.push("completed = ?"); values.push(input.completed ? 1 : 0); }
  if (input.priority !== undefined) { fields.push("priority = ?"); values.push(input.priority); }
  if (input.due_date !== undefined) { fields.push("due_date = ?"); values.push(input.due_date); }

  if (fields.length === 0) return existing;

  fields.push("updated_at = datetime('now')");
  values.push(id, userId);

  db.prepare(`UPDATE todos SET ${fields.join(", ")} WHERE id = ? AND user_id = ?`).run(...values);
  return findById(id, userId);
}

export function remove(id: number, userId: number): boolean {
  const db = getDatabase();
  const result = db.prepare("DELETE FROM todos WHERE id = ? AND user_id = ?").run(id, userId);
  return result.changes > 0;
}

function normalizeTodo(row: Todo): Todo {
  return { ...row, completed: Boolean(row.completed) };
}
