import type { ConversationTurn } from './types.js';

/**
 * Build a chat prompt string from conversation history with an optional
 * system instruction and optional tool declarations.
 *
 * Gemma 4 expresses system instructions through the `developer` role. When
 * `systemPrompt` is provided, its text is emitted first inside a
 * `<start_of_turn>developer ... <end_of_turn>` block. If `toolsJson` is
 * also provided, the function-calling preamble + declarations are appended
 * to the same developer turn so the model sees one coherent preamble.
 */
export function buildChatPrompt(
  history: ConversationTurn[],
  toolsJson?: string,
  systemPrompt?: string | null,
): string {
  let tools: unknown[] | null = null;
  try {
    if (toolsJson) {
      const parsed = JSON.parse(toolsJson);
      if (Array.isArray(parsed) && parsed.length > 0) tools = parsed;
    }
  } catch {
    tools = null;
  }

  const trimmedSystem = (systemPrompt ?? '').trim();
  let developerBody = '';
  if (trimmedSystem) developerBody += trimmedSystem + '\n';

  if (tools) {
    let declarations = '';
    for (const tool of tools as Array<{
      name: string;
      description: string;
      parameters?: {
        type: string;
        properties?: Record<string, { description: string; type: string; enum?: string[] }>;
        required?: string[];
      };
    }>) {
      declarations += `<start_function_declaration>declaration:${tool.name}{`;
      declarations += `description:<escape>${tool.description}<escape>`;
      if (tool.parameters) {
        declarations += `,parameters:{properties:{`;
        const props = Object.entries(tool.parameters.properties || {});
        declarations += props.map(([k, v]) => {
          let param = `${k}:{description:<escape>${v.description}<escape>,type:<escape>${v.type}<escape>`;
          if (v.enum) {
            param += `,enum:[${v.enum.map((e) => `<escape>${e}<escape>`).join(',')}]`;
          }
          param += '}';
          return param;
        }).join(',');
        declarations += `}`;
        if (tool.parameters.required) {
          declarations += `,required:[${tool.parameters.required.map((r) => `<escape>${r}<escape>`).join(',')}]`;
        }
        declarations += `,type:<escape>${tool.parameters.type}<escape>`;
        declarations += `}`;
      }
      declarations += `}<end_function_declaration>`;
    }
    developerBody += `You are a model that can do function calling with the following functions\n${declarations}\n`;
  }

  let prompt = '';
  if (developerBody) {
    prompt += `<start_of_turn>developer\n${developerBody.trimEnd()}<end_of_turn>\n`;
  }
  for (const turn of history) {
    prompt += `<start_of_turn>${turn.role}\n${turn.text}<end_of_turn>\n`;
  }
  prompt += `<start_of_turn>model\n`;
  return prompt;
}
