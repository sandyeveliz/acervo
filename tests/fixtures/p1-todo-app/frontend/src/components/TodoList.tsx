import { useState } from "react";
import { TodoItem } from "./TodoItem";
import { TodoForm } from "./TodoForm";
import type { Todo, TodoFilters, CreateTodoRequest } from "../types/todo";
import "../styles/todo.css";

interface TodoListProps {
  todos: Todo[];
  loading: boolean;
  error: string | null;
  onAdd: (input: CreateTodoRequest) => Promise<void>;
  onToggle: (id: number) => Promise<void>;
  onEdit: (id: number, input: Partial<Todo>) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
  onFilterChange: (filters: TodoFilters) => void;
}

export function TodoList({ todos, loading, error, onAdd, onToggle, onEdit, onDelete, onFilterChange }: TodoListProps) {
  const [showForm, setShowForm] = useState(false);
  const [filter, setFilter] = useState<"all" | "active" | "completed">("all");

  const handleFilterChange = (newFilter: "all" | "active" | "completed") => {
    setFilter(newFilter);
    onFilterChange({
      completed: newFilter === "all" ? undefined : newFilter === "completed",
    });
  };

  const handleAdd = async (input: CreateTodoRequest) => {
    await onAdd(input);
    setShowForm(false);
  };

  if (loading) {
    return <div className="todo-list-loading">Loading todos...</div>;
  }

  if (error) {
    return <div className="todo-list-error">Error: {error}</div>;
  }

  return (
    <div className="todo-list">
      <div className="todo-list-header">
        <h2>My Todos ({todos.length})</h2>
        <button className="btn btn-primary" onClick={() => setShowForm(!showForm)}>
          {showForm ? "Cancel" : "Add Todo"}
        </button>
      </div>

      {showForm && <TodoForm onSubmit={handleAdd} />}

      <div className="todo-filters">
        {(["all", "active", "completed"] as const).map((f) => (
          <button
            key={f}
            className={`btn btn-filter ${filter === f ? "active" : ""}`}
            onClick={() => handleFilterChange(f)}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {todos.length === 0 ? (
        <div className="todo-list-empty">
          <p>No todos yet. Create one to get started!</p>
        </div>
      ) : (
        <div className="todo-items">
          {todos.map((todo) => (
            <TodoItem
              key={todo.id}
              todo={todo}
              onToggle={() => onToggle(todo.id)}
              onDelete={() => onDelete(todo.id)}
              onEdit={() => onEdit(todo.id, {})}
            />
          ))}
        </div>
      )}
    </div>
  );
}
