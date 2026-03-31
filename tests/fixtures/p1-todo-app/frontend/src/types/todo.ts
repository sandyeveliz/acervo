export interface Todo {
  id: number;
  user_id: number;
  title: string;
  description: string;
  completed: boolean;
  priority: Priority;
  due_date: string | null;
  created_at: string;
  updated_at: string;
}

export type Priority = "low" | "medium" | "high";

export interface CreateTodoRequest {
  title: string;
  description?: string;
  priority?: Priority;
  due_date?: string;
}

export interface UpdateTodoRequest {
  title?: string;
  description?: string;
  completed?: boolean;
  priority?: Priority;
  due_date?: string | null;
}

export interface TodoFilters {
  completed?: boolean;
  priority?: Priority;
  search?: string;
}

export interface TodoStats {
  total: number;
  completed: number;
  pending: number;
  completionRate: number;
}

export interface ApiResponse<T> {
  data: T;
  count?: number;
}

export interface ApiError {
  error: string;
}

export interface AuthUser {
  id: number;
  email: string;
  name: string;
}

export interface AuthResult {
  user: AuthUser;
  token: string;
}
