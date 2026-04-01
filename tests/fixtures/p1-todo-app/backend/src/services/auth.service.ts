import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import * as UserModel from "../models/user.model";
import { config } from "../config/env";

const SALT_ROUNDS = 10;

export interface AuthResult {
  user: { id: number; email: string; name: string };
  token: string;
}

export class AuthService {
  async register(email: string, password: string, name: string): Promise<AuthResult> {
    const existing = UserModel.findByEmail(email);
    if (existing) {
      throw new AuthError("Email already registered");
    }

    if (password.length < 8) {
      throw new AuthError("Password must be at least 8 characters");
    }

    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
    const user = UserModel.create({ email, password_hash: passwordHash, name });

    const token = this.generateToken(user.id);
    return {
      user: { id: user.id, email: user.email, name: user.name },
      token,
    };
  }

  async login(email: string, password: string): Promise<AuthResult> {
    const user = UserModel.findByEmail(email);
    if (!user) {
      throw new AuthError("Invalid email or password");
    }

    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) {
      throw new AuthError("Invalid email or password");
    }

    const token = this.generateToken(user.id);
    return {
      user: { id: user.id, email: user.email, name: user.name },
      token,
    };
  }

  verifyToken(token: string): { userId: number } {
    try {
      const payload = jwt.verify(token, config.jwtSecret) as { sub: number };
      return { userId: payload.sub };
    } catch {
      throw new AuthError("Invalid or expired token");
    }
  }

  private generateToken(userId: number): string {
    return jwt.sign({ sub: userId }, config.jwtSecret, {
      expiresIn: config.jwtExpiresIn,
    });
  }
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AuthError";
  }
}
