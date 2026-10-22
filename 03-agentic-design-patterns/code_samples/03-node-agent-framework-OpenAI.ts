import { OpenAI } from "openai";

import { toResponseInputItems } from "openai/lib/responses/ResponseInputItems";

type OpenAIResponseInput = OpenAI.Responses.ResponseInput;

declare const process: {
  env: Record<string, string | undefined>;
};

// ============================================================================
// AGENTIC DESIGN PRINCIPLES DEMONSTRATION
// ============================================================================
// This sample demonstrates the three key design principles from the lesson:
// 1. TRANSPARENCY: The agent explains what it's doing and why
// 2. CONTROL: Users can customize preferences and the agent respects them
// 3. CONSISTENCY: The agent uses a predictable, standardized interaction pattern
// ============================================================================

// Tool Function: Random Destination Generator
// TRANSPARENCY: Clear description helps users understand this tool's purpose

const destinations = [
  "Paris, France",
  "Tokyo, Japan",
  "New York City, USA",
  "Sydney, Australia",
  "Rome, Italy",
  "Barcelona, Spain",
  "Cape Town, South Africa",
  "Rio de Janeiro, Brazil",
  "Bangkok, Thailand",
  "Vancouver, Canada",
];

// Provides a random vacation destination.
function GetRandomDestination(): string {
  const randomIndex = Math.floor(Math.random() * destinations.length);

  return destinations[randomIndex];
}

// Saves user preferences for trip planning. Use this when the user specifies preferences like budget level (budget/moderate/luxury),
// trip style (adventure/relaxation/cultural), or duration preference.
function SaveUserPreferences(preferenceType: string, preferenceValue: string): string {
  // CONTROL: Save user preferences for future interactions
  // In a real application, this could be saved to a database or file
  console.log(`\n[TRANSPARENCY] Saving preference: ${preferenceType} = ${preferenceValue}`);
  return `Preference saved: ${preferenceType} is now set to '${preferenceValue}'. I will remember this for future suggestions.`;
}

// Declare tools
const tools: OpenAI.Responses.Tool[] = [
  {
    type: "function",
    name: "GetRandomDestination",
    description: "Provides a random vacation destination.",
    parameters: {},
    strict: true,
  },
  {
    type: "function",
    name: "SaveUserPreferences",
    description: "Saves user preferences for trip planning (e.g., budget, trip style, duration).",
    parameters: {
      type: "object",
      properties: {
        preferenceType: { type: "string", description: "The type of preference being saved, e.g., 'budget', 'style', 'duration'" },
        preferenceValue: { type: "string", description: "The value of the preference" },
      },
      required: ["preferenceType", "preferenceValue"],
    },
    strict: true,
  },
];

const openRouterEndpoint = "https://openrouter.ai/api/v1";

const model = "openrouter/free";

const apiKey = process.env["OPENROUTER_API_KEY"] ?? (() => { throw new Error("OPENROUTER_API_KEY is not set."); })();

const client = new OpenAI({
  apiKey: apiKey,
  baseURL: openRouterEndpoint,
});

// ============================================================================
// AGENT INSTRUCTIONS - Demonstrating Design Principles
// ============================================================================
// These instructions embed the three design principles directly into the agent's behavior
const instruction = `
You are a helpful AI Agent that demonstrates the Agentic Design Principles.

## Your Core Principles

**TRANSPARENCY**: Always explain what you're doing and why.
- When using a tool, briefly explain which tool you're calling and why
- Share your reasoning process with the user
- Be honest about limitations or uncertainties

**CONTROL**: Respect user preferences and allow customization.
- Ask about preferences before making assumptions
- Use the SaveUserPreference tool to remember user choices
- Always prioritize explicit user requests over defaults

**CONSISTENCY**: Use a predictable, standardized interaction pattern.
- Start every conversation with a friendly greeting
- Structure responses in a clear, organized format
- Use similar phrasing for similar actions

## Initial Greeting (CONSISTENCY)

When the conversation begins, always introduce yourself with this message:
"Hello! I'm TravelAgent, your AI vacation planning assistant.

🔍 **Transparency**: I'll always explain my reasoning and the tools I use.
🎮 **Control**: Tell me your preferences, and I'll remember them.
🔄 **Consistency**: I follow a predictable pattern to make planning easy.

What kind of trip would you like me to help you plan today?"

## Guidelines
- When users specify a destination, plan for that location
- Only suggest random destinations when the user hasn't specified one
- Always confirm before making changes to preferences
`;

// Create a running input list we will add to over time
let inputList: OpenAIResponseInput = [
  { role: "user", content: "Hello" },
];

// ============================================================================
// DEMONSTRATION: Start with "Hello" to trigger the greeting (Issue #402 fix)
// ============================================================================
console.log("=== Demonstrating Agentic Design Principles ===\n");
console.log("User: Hello\n");
console.log("Agent Response:");

inputList = await CreateOpenAIResponse(inputList);

console.log("\n");

// ============================================================================
// DEMONSTRATION: User sets a preference (CONTROL principle)
// ============================================================================
console.log("---");
console.log("User: I prefer luxury travel and cultural experiences.\n");
console.log("Agent Response:");

inputList.push({ role: "user", content: "I prefer luxury travel and cultural experiences." });
inputList = await CreateOpenAIResponse(inputList);

console.log("\n");

// ============================================================================
// DEMONSTRATION: Agent uses tools with transparency
// ============================================================================
console.log("---");
console.log("User: Suggest a destination for me.\n");
console.log("Agent Response:");

inputList.push({ role: "user", content: "Suggest a destination for me." });
inputList = await CreateOpenAIResponse(inputList);

// Helper function
async function CreateOpenAIResponse(inputList: OpenAIResponseInput): Promise<OpenAIResponseInput> {
  let requiresAction = false;

  do {
    requiresAction = false;

    let response = await client.responses.create({
      model: model,
      instructions: instruction,
      input: inputList,
      tools: tools,
    });

    // Preserve model output for the next turn
    inputList.push(...toResponseInputItems(response.output));

    for (const item of response.output) {
      switch (item.type) {
        case "message":
          handleMessageResponse(item);
          break;
        case "reasoning":
          handleReasonResponse(item);
          break;
        case "function_call":
          requiresAction = true;
          handleFunctionCallResponse(item, inputList);
          break;
      }
    }
  } while (requiresAction);

  return inputList;
}

function handleFunctionCallResponse(item: OpenAI.Responses.ResponseFunctionToolCall, inputList: OpenAIResponseInput) {
  console.log(
    `ToolCall:\n${item.name} with param: ${JSON.stringify(item.arguments)}`
  );
  if (item.name === "GetRandomDestination") {
    // Execute the function logic for GetRandomDestination
    handleGetRandomDestinationToolCall(inputList, item);
  } else if (item.name === "SaveUserPreferences") {
    handleSaveUserPreferencesToolCall(item, inputList);
  }
}

function handleSaveUserPreferencesToolCall(item: OpenAI.Responses.ResponseFunctionToolCall, inputList: OpenAI.Responses.ResponseInput) {
  const toolArguments = JSON.parse(item.arguments);
  const preference = SaveUserPreferences(toolArguments.preferenceType, toolArguments.preferenceValue);
  inputList.push({
    type: "function_call_output",
    call_id: item.call_id,
    output: preference,
  });

  console.log(`ToolCallResult:\n${preference}`);
}

function handleGetRandomDestinationToolCall(inputList: OpenAI.Responses.ResponseInput, item: OpenAI.Responses.ResponseFunctionToolCall) {
  const destination = GetRandomDestination();
  // Provide function call results to the model
  inputList.push({
    type: "function_call_output",
    call_id: item.call_id,
    output: destination,
  });

  console.log(`ToolCallResult:\n${destination}`);
}

function handleReasonResponse(item: OpenAI.Responses.ResponseReasoningItem) {
  console.log(`REASONING:`);
  item.content?.forEach((contentItem) => {
    console.log(contentItem.text);
  });
}

function handleMessageResponse(item: OpenAI.Responses.ResponseOutputMessage) {
  console.log(`${item.role.toUpperCase()}:`);
  item.content.forEach((contentItem) => {
    if (contentItem.type === "output_text") {
      console.log(`${contentItem.text}`);
    } else if (contentItem.type === "refusal") {
      console.log(`${contentItem.refusal}`);
    }
  });
}
