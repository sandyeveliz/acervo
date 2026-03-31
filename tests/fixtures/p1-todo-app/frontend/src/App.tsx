import { useState } from "react";
import { Header } from "./components/Header";
import { TodoList } from "./components/TodoList";
import { useTodos } from "./hooks/useTodos";
import { useAuth } from "./hooks/useAuth";
import type { TodoFilters } from "./types/todo";
import "./styles/app.css";

export default function App() {
  const { user, authenticated, login, register, logout } = useAuth();
  const [filters, setFilters] = useState<TodoFilters>({});
  const { todos, loading, error, addTodo, toggleTodo, editTodo, removeTodo } = useTodos(filters);

  if (!authenticated) {
    return (
      <div className="app">
        <Header user={null} onLogout={() => {}} />
        <main className="auth-page">
          <h2>Sign in to manage your todos</h2>
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              const form = e.target as HTMLFormElement;
              const email = (form.elements.namedItem("email") as HTMLInputElement).value;
              const password = (form.elements.namedItem("password") as HTMLInputElement).value;
              await login(email, password);
            }}
          >
            <input name="email" type="email" placeholder="Email" required />
            <input name="password" type="password" placeholder="Password" required />
            <button type="submit" className="btn btn-primary">Login</button>
          </form>
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <Header user={user} onLogout={logout} />
      <main className="main-content">
        <TodoList
          todos={todos}
          loading={loading}
          error={error}
          onAdd={addTodo}
          onToggle={toggleTodo}
          onEdit={editTodo}
          onDelete={removeTodo}
          onFilterChange={setFilters}
        />
      </main>
    </div>
  );
}
