import json
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class Llama2Wrapper:
    def __init__(self, client=None) -> None:
        self.client = client

    def invoke_with_text(self, prompt):
        # init llama2 runtime client
        client = self.client or boto3.client(
            service_name = "bedrock-runtime",
            region_name = "us-east-1"
        )

        # invoke llama2 with the text prompt
        model_id = "meta.llama2-13b-chat-v1"

        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "prompt": prompt,
                        "max_gen_len": 512,
                        "temperature": 0.5,
                        "top_p": 0.9,
                    }
                ),
            )

            # process and print the response
            result = json.loads(response.get("body").read())
            print(result)
            """
            input_tokens = result["inputTextTokenCount"]
            output_tokens = 0
            output_list = result.get("results", [])

            for output in output_list:
                output_tokens += output["tokenCount"]

            print("Invocation details:")
            print(f"- The input length is {input_tokens} tokens.")
            print(f"- The output length is {output_tokens} tokens.")

            print(f"- The model returned {len(output_list)} response(s):")
            for output in output_list:
                print(output["outputText"])
                """
            return result

        except ClientError as err:
            logger.error(
                "Couldn't invoke Llama 2 Chat 13B. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise

def usage_demo():
    print("-" * 88)
    print("Welcome to the Amazon Bedrock Runtime demo with Llama 2 Chat 13B.")
    print("-" * 88)

    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    wrapper = Llama2Wrapper(client)

    # invoke Llama 2 Chat 13B with a text prompt
    text_prompt = "Hi, write a short sentence about yourself"
    print(f"Invoking Llama 2 Chat 13B with '{text_prompt}'...")
    wrapper.invoke_with_text(text_prompt)
    print("-" * 88)

if __name__ == "__main__":
    usage_demo()
