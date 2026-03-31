import * as TodoModel from "../models/todo.model";
import type { CreateTodoInput, UpdateTodoInput, TodoFilters, Todo } from "../models/todo.model";

export class TodoService {
  async listTodos(userId: number, filters: TodoFilters = {}): Promise<Todo[]> {
    return TodoModel.findAllByUser(userId, filters);
  }

  async getTodo(id: number, userId: number): Promise<Todo | null> {
    return TodoModel.findById(id, userId);
  }

  async createTodo(userId: number, input: CreateTodoInput): Promise<Todo> {
    if (!input.title || input.title.trim().length === 0) {
      throw new ValidationError("Title is required");
    }
    if (input.title.length > 200) {
      throw new ValidationError("Title must be 200 characters or less");
    }
    if (input.due_date) {
      const date = new Date(input.due_date);
      if (isNaN(date.getTime())) {
        throw new ValidationError("Invalid due date format");
      }
    }

    return TodoModel.create(userId, {
      ...input,
      title: input.title.trim(),
      description: input.description?.trim(),
    });
  }

  async updateTodo(id: number, userId: number, input: UpdateTodoInput): Promise<Todo> {
    const existing = TodoModel.findById(id, userId);
    if (!existing) {
      throw new NotFoundError(`Todo ${id} not found`);
    }

    if (input.title !== undefined && input.title.trim().length === 0) {
      throw new ValidationError("Title cannot be empty");
    }

    const updated = TodoModel.update(id, userId, input);
    return updated!;
  }

  async deleteTodo(id: number, userId: number): Promise<void> {
    const deleted = TodoModel.remove(id, userId);
    if (!deleted) {
      throw new NotFoundError(`Todo ${id} not found`);
    }
  }

  async getStats(userId: number): Promise<{ total: number; completed: number; pending: number; completionRate: number }> {
    const todos = TodoModel.findAllByUser(userId);
    const completed = todos.filter(t => t.completed).length;
    const total = todos.length;
    return {
      total,
      completed,
      pending: total - completed,
      completionRate: total > 0 ? Math.round((completed / total) * 100) : 0,
    };
  }
}

export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValidationError";
  }
}

export class NotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NotFoundError";
  }
}
