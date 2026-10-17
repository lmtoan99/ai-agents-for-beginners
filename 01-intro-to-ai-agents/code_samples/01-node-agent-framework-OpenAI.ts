import { OpenAI } from "openai";

import { toResponseInputItems } from "openai/lib/responses/ResponseInputItems";

type OpenAIResponseInput = OpenAI.Responses.ResponseInput;

declare const process: {
  env: Record<string, string | undefined>;
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

inputList = await CreateOpenAIResponse(inputList);


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
  if (item.name === "GetRandomDestination") {
    console.log(
      `ToolCall:\n${item.name} with param: ${JSON.stringify(item.arguments)}`
    );

    // Execute the function logic for GetRandomDestination
    const destination = GetRandomDestination();

    // Provide function call results to the model
    inputList.push({
      type: "function_call_output",
      call_id: item.call_id,
      output: destination,
    });

    console.log(`ToolCallResult:\n${destination}`);
  }
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

