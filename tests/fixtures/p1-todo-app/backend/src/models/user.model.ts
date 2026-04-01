import { getDatabase } from "../config/database";

export interface User {
  id: number;
  email: string;
  password_hash: string;
  name: string;
  created_at: string;
}

export interface CreateUserInput {
  email: string;
  password_hash: string;
  name: string;
}

export function findByEmail(email: string): User | null {
  const db = getDatabase();
  return (db.prepare("SELECT * FROM users WHERE email = ?").get(email) as User) || null;
}

export function findById(id: number): User | null {
  const db = getDatabase();
  return (db.prepare("SELECT * FROM users WHERE id = ?").get(id) as User) || null;
}

export function create(input: CreateUserInput): User {
  const db = getDatabase();
  const result = db.prepare(
    "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)"
  ).run(input.email, input.password_hash, input.name);

  return findById(result.lastInsertRowid as number)!;
}

export function countTodos(userId: number): { total: number; completed: number; pending: number } {
  const db = getDatabase();
  const row = db.prepare(`
    SELECT
      COUNT(*) as total,
      SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) as completed,
      SUM(CASE WHEN completed = 0 THEN 1 ELSE 0 END) as pending
    FROM todos WHERE user_id = ?
  `).get(userId) as { total: number; completed: number; pending: number };

  return row;
}
