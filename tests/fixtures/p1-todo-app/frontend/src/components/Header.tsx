import type { AuthUser } from "../types/todo";
import "../styles/header.css";

interface HeaderProps {
  user: AuthUser | null;
  onLogout: () => void;
}

export function Header({ user, onLogout }: HeaderProps) {
  return (
    <header className="header">
      <div className="header-left">
        <h1 className="header-title">Todo App</h1>
      </div>
      {user && (
        <div className="header-right">
          <span className="header-user">Welcome, {user.name}</span>
          <button className="btn btn-outline" onClick={onLogout}>
            Logout
          </button>
        </div>
      )}
    </header>
  );
}
