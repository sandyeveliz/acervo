import type { Todo, Priority } from "../types/todo";
import "../styles/todo.css";

interface TodoItemProps {
  todo: Todo;
  onToggle: (id: number) => void;
  onDelete: (id: number) => void;
  onEdit: (id: number) => void;
}

const priorityLabels: Record<Priority, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
};

const priorityColors: Record<Priority, string> = {
  low: "priority-low",
  medium: "priority-medium",
  high: "priority-high",
};

export function TodoItem({ todo, onToggle, onDelete, onEdit }: TodoItemProps) {
  const isOverdue = todo.due_date && new Date(todo.due_date) < new Date() && !todo.completed;

  return (
    <div className={`todo-item ${todo.completed ? "todo-completed" : ""} ${isOverdue ? "todo-overdue" : ""}`}>
      <div className="todo-checkbox">
        <input
          type="checkbox"
          checked={todo.completed}
          onChange={() => onToggle(todo.id)}
          aria-label={`Mark "${todo.title}" as ${todo.completed ? "incomplete" : "complete"}`}
        />
      </div>

      <div className="todo-content">
        <span className="todo-title">{todo.title}</span>
        {todo.description && <p className="todo-description">{todo.description}</p>}
        <div className="todo-meta">
          <span className={`todo-priority ${priorityColors[todo.priority]}`}>
            {priorityLabels[todo.priority]}
          </span>
          {todo.due_date && (
            <span className="todo-due-date">
              Due: {new Date(todo.due_date).toLocaleDateString()}
            </span>
          )}
        </div>
      </div>

      <div className="todo-actions">
        <button className="btn btn-icon" onClick={() => onEdit(todo.id)} aria-label="Edit todo">
          Edit
        </button>
        <button className="btn btn-icon btn-danger" onClick={() => onDelete(todo.id)} aria-label="Delete todo">
          Delete
        </button>
      </div>
    </div>
  );
}
