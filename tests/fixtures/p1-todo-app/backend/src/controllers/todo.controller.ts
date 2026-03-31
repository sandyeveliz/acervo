import { Request, Response, NextFunction } from "express";
import { TodoService, ValidationError, NotFoundError } from "../services/todo.service";

const todoService = new TodoService();

export async function listTodos(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const userId = req.userId!;
    const filters = {
      completed: req.query.completed !== undefined ? req.query.completed === "true" : undefined,
      priority: req.query.priority as "low" | "medium" | "high" | undefined,
      search: req.query.search as string | undefined,
    };

    const todos = await todoService.listTodos(userId, filters);
    res.json({ data: todos, count: todos.length });
  } catch (err) {
    next(err);
  }
}

export async function getTodo(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const userId = req.userId!;
    const todoId = parseInt(req.params.id, 10);

    const todo = await todoService.getTodo(todoId, userId);
    if (!todo) {
      res.status(404).json({ error: "Todo not found" });
      return;
    }

    res.json({ data: todo });
  } catch (err) {
    next(err);
  }
}

export async function createTodo(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const userId = req.userId!;
    const todo = await todoService.createTodo(userId, req.body);
    res.status(201).json({ data: todo });
  } catch (err) {
    if (err instanceof ValidationError) {
      res.status(400).json({ error: err.message });
      return;
    }
    next(err);
  }
}

export async function updateTodo(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const userId = req.userId!;
    const todoId = parseInt(req.params.id, 10);
    const todo = await todoService.updateTodo(todoId, userId, req.body);
    res.json({ data: todo });
  } catch (err) {
    if (err instanceof ValidationError) {
      res.status(400).json({ error: err.message });
      return;
    }
    if (err instanceof NotFoundError) {
      res.status(404).json({ error: err.message });
      return;
    }
    next(err);
  }
}

export async function deleteTodo(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const userId = req.userId!;
    const todoId = parseInt(req.params.id, 10);
    await todoService.deleteTodo(todoId, userId);
    res.status(204).send();
  } catch (err) {
    if (err instanceof NotFoundError) {
      res.status(404).json({ error: err.message });
      return;
    }
    next(err);
  }
}

export async function getStats(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const userId = req.userId!;
    const stats = await todoService.getStats(userId);
    res.json({ data: stats });
  } catch (err) {
    next(err);
  }
}
