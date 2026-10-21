import { OpenAI } from "openai";

import { toResponseInputItems } from "openai/lib/responses/ResponseInputItems";

type OpenAIResponseInput = OpenAI.Responses.ResponseInput;

declare const process: {
  env: Record<string, string | undefined>;
  stdout: {
    write: (message: string) => void;
  };
};

// Tool Function: Random Destination Generator

// This static method will be available to the agent as a callable tool

// The [Description] attribute helps the AI understand when to use this function

// This demonstrates how to create custom tools for AI agents

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

// Declare tools

const tools: OpenAI.Responses.Tool[] = [
  {
    type: "function",
    name: "GetRandomDestination",
    description: "Provides a random vacation destination.",
    parameters: {},
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

const instruction =
  "You are a helpful AI Agent that can help plan vacations for customers at random destinations";

// Create a running input list we will add to over time

let inputList: OpenAIResponseInput = [
  { role: "user", content: "Plan me a day trip" },
];

// Execute Agent: Plan a Day Trip

// Run the agent with streaming enabled for real-time response display

// Shows the agent's thinking and response as it generates the content

// Provides better user experience with immediate feedback

// Print system and user messages
console.log("SYSTEM:\n" + instruction);
console.log("USER:\nPlan me a day trip");

inputList = await CreateOpenAIStreamResponse(inputList);

console.log();

// Execute Agent: Follow-up Request with Context Awareness
// Demonstrate contextual conversation by referencing the previous response
// The agent remembers the previous destination suggestion and will provide an alternative
// This showcases the power of conversation sessions and contextual understanding
// inputList.push({ role: "user", content: "I don't like that destination. Plan me another vacation." });
// console.log("USER:\nI don't like that destination. Plan me another vacation.");
// inputList = await CreateOpenAIResponse(inputList);

// Function to create an OpenAI response and handle the output
async function CreateOpenAIStreamResponse(inputList: OpenAIResponseInput): Promise<OpenAIResponseInput> {
  let requiresAction = false;

  do {
    requiresAction = false;

    let stream = await client.responses.create({
      stream: true,
      model: model,
      instructions: instruction,
      input: inputList,
      tools: tools,
    });

    const extraMessages: OpenAI.Responses.ResponseInputItem[] = [];

    for await (const event of stream) {
      if (event.type === "response.content_part.added") {
        if (event.part.type == "output_text") {
          process.stdout.write(`Assistant:\n${event.part.text}`);
        } else if (event.part.type == "reasoning_text") {
          process.stdout.write(`Reasoning:\n${event.part.text}`);
        } else if (event.part.type == "refusal") {
          process.stdout.write(`Refusal:\n${event.part.refusal}`);
        }
      }
      else if (event.type === "response.output_text.delta") {
        process.stdout.write(`${event.delta}`);
      }
      else if (event.type === "response.reasoning_text.delta") {
        process.stdout.write(`${event.delta}`);
      }
      else if (event.type === "response.refusal.delta") {
        process.stdout.write(`${event.delta}`);
      }
      else if (event.type === "response.output_text.done") {
        console.log("\n-------------End of output text");
      }
      else if (event.type === "response.reasoning_text.done") {
        console.log("\n-------------End of reasoning text");
      }
      else if (event.type === "response.refusal.done") {
        console.log("\n-------------End of refusal text");
      }
      else if (event.type === "response.output_item.added") {
        if (event.item.type === "function_call") {
          console.log(`Function call: ${event.item.name} with args: ${JSON.stringify(event.item.arguments)}`);
          requiresAction = true;
          // Execute the function logic for GetRandomDestination
          const destination = GetRandomDestination();

          // Provide function call results to the model
          extraMessages.push({
            type: "function_call_output",
            call_id: event.item.call_id,
            output: destination,
          });

          console.log(`ToolCallResult:\n${destination}`);
        }
      }
      else if (event.type === "response.function_call_arguments.done") [
        console.log(`\n-------------Function call arguments done --- ${event.item_id} --- ${event.arguments}`)
      ]
      else if (event.type === "response.completed") {
        console.log("\n-------------Response completed");
        // Preserve model output for the next turn
        inputList.push(...toResponseInputItems(event.response.output));
      }
    }
    // Add tool call responses
    inputList.push(...extraMessages);
  } while (requiresAction);

  return inputList;
}
