import type { ApiResponse, ApiError, Todo, CreateTodoRequest, UpdateTodoRequest, TodoFilters, TodoStats, AuthResult } from "../types/todo";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:3001/api";

function getToken(): string | null {
  return localStorage.getItem("auth_token");
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) || {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as ApiError;
    throw new ApiRequestError(body.error || `HTTP ${response.status}`, response.status);
  }

  if (response.status === 204) return undefined as T;
  return response.json();
}

// Auth
export async function login(email: string, password: string): Promise<AuthResult> {
  const res = await request<ApiResponse<AuthResult>>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  localStorage.setItem("auth_token", res.data.token);
  return res.data;
}

export async function register(email: string, password: string, name: string): Promise<AuthResult> {
  const res = await request<ApiResponse<AuthResult>>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
  localStorage.setItem("auth_token", res.data.token);
  return res.data;
}

export function logout(): void {
  localStorage.removeItem("auth_token");
}

export function isAuthenticated(): boolean {
  return !!getToken();
}

// Todos
export async function fetchTodos(filters?: TodoFilters): Promise<Todo[]> {
  const params = new URLSearchParams();
  if (filters?.completed !== undefined) params.set("completed", String(filters.completed));
  if (filters?.priority) params.set("priority", filters.priority);
  if (filters?.search) params.set("search", filters.search);

  const query = params.toString();
  const res = await request<ApiResponse<Todo[]>>(`/todos${query ? `?${query}` : ""}`);
  return res.data;
}

export async function fetchTodo(id: number): Promise<Todo> {
  const res = await request<ApiResponse<Todo>>(`/todos/${id}`);
  return res.data;
}

export async function createTodo(input: CreateTodoRequest): Promise<Todo> {
  const res = await request<ApiResponse<Todo>>("/todos", {
    method: "POST",
    body: JSON.stringify(input),
  });
  return res.data;
}

export async function updateTodo(id: number, input: UpdateTodoRequest): Promise<Todo> {
  const res = await request<ApiResponse<Todo>>(`/todos/${id}`, {
    method: "PUT",
    body: JSON.stringify(input),
  });
  return res.data;
}

export async function deleteTodo(id: number): Promise<void> {
  await request(`/todos/${id}`, { method: "DELETE" });
}

export async function fetchTodoStats(): Promise<TodoStats> {
  const res = await request<ApiResponse<TodoStats>>("/todos/stats");
  return res.data;
}

export class ApiRequestError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "ApiRequestError";
  }
}
