// src/services/chatService.ts

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  user_id?: string;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  sources: Array<{
    title: string;
    score: number;
    content_preview: string;
  }>;
  timestamp: Date;
}

export interface ConversationHistoryRequest {
  conversation_id: string;
  limit?: number;
}

export interface Message {
  role: string;
  content: string;
  timestamp: Date;
}

export interface ConversationHistoryResponse {
  messages: Message[];
  conversation_id: string;
}

export interface ResetResponse {
  conversation_id: string;
  message: string;
}

export class ChatService {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || process.env.BACKEND_URL || 'http://localhost:8000';
  }

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: request.message,
          conversation_id: request.conversation_id,
          user_id: request.user_id || 'docusaurus-user',
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Convert timestamp string to Date object if needed
      return {
        ...data,
        timestamp: new Date(data.timestamp)
      };
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }

  async getConversationHistory(request: ConversationHistoryRequest): Promise<ConversationHistoryResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/chat/history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conversation_id: request.conversation_id,
          limit: request.limit || 10,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Convert timestamp strings to Date objects if needed
      return {
        ...data,
        messages: data.messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
      };
    } catch (error) {
      console.error('Error getting conversation history:', error);
      throw error;
    }
  }

  async resetConversation(): Promise<ResetResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/chat/reset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error resetting conversation:', error);
      throw error;
    }
  }
}

// Default instance for backward compatibility
export const chatService = new ChatService();