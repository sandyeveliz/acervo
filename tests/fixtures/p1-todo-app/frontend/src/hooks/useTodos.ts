import { useState, useEffect, useCallback } from "react";
import type { Todo, CreateTodoRequest, UpdateTodoRequest, TodoFilters } from "../types/todo";
import { fetchTodos, createTodo, updateTodo, deleteTodo } from "../services/api";

interface UseTodosResult {
  todos: Todo[];
  loading: boolean;
  error: string | null;
  addTodo: (input: CreateTodoRequest) => Promise<void>;
  toggleTodo: (id: number) => Promise<void>;
  editTodo: (id: number, input: UpdateTodoRequest) => Promise<void>;
  removeTodo: (id: number) => Promise<void>;
  refresh: () => Promise<void>;
}

export function useTodos(filters?: TodoFilters): UseTodosResult {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchTodos(filters);
      setTodos(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch todos");
    } finally {
      setLoading(false);
    }
  }, [filters?.completed, filters?.priority, filters?.search]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const addTodo = useCallback(async (input: CreateTodoRequest) => {
    const newTodo = await createTodo(input);
    setTodos(prev => [newTodo, ...prev]);
  }, []);

  const toggleTodo = useCallback(async (id: number) => {
    const todo = todos.find(t => t.id === id);
    if (!todo) return;

    const updated = await updateTodo(id, { completed: !todo.completed });
    setTodos(prev => prev.map(t => (t.id === id ? updated : t)));
  }, [todos]);

  const editTodo = useCallback(async (id: number, input: UpdateTodoRequest) => {
    const updated = await updateTodo(id, input);
    setTodos(prev => prev.map(t => (t.id === id ? updated : t)));
  }, []);

  const removeTodo = useCallback(async (id: number) => {
    await deleteTodo(id);
    setTodos(prev => prev.filter(t => t.id !== id));
  }, []);

  return { todos, loading, error, addTodo, toggleTodo, editTodo, removeTodo, refresh };
}
