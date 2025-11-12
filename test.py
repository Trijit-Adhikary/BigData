import os
from openai import AzureOpenAI
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

results = []
lessdata_columns = lessdata.columns

for row in lessdata.collect():
    text_content = getattr(row, 'text_content', None)
    abstract_text = getattr(row, 'ABSTACT_TEXT', None)
    use_abstract = (
        text_content is None or
        (isinstance(text_content, str) and (
            text_content.strip() == "" or
            text_content.strip() == "The original PDF is Password Protected for Opening. Unable to extract text for Index."
        ))
    )
    if not use_abstract:
        prompt_template = f"""
        ### Role
        You are an AI model that classifies document text into the most relevant categories strictly from provided reference lists.

        ### Instructions
        1. Always return output as a JSON object with key-value pairs.
        2. The value for 'Relevant Topic' and 'Relevant Resource Type' should be a string.
        3. The values for 'Relevant Outcome Area', 'Relevant Enterprise Topic', and 'Relevant Keywords' should be lists.
        4. If an exact match is not available, select the closest existing concept in the list.
        5. Do not change or modify the given input values while assigning the values.
        6. Do not add any additional text to the output.Make sure there is no markdown or HTML tags are allowed in the output.
        7. Output format must follow the structure below exactly:

        {{
            "Relevant Topic": "string",
            "Relevant Resource Type": "string",
            "Relevant Outcome Area": ["item1", "item2", "item3"],
            "Relevant Enterprise Topic": ["item1", "item2", ..., "item10"],
            "Relevant Keywords": ["item1", "item2", ..., "item10"]
        }}

        ### Input
        Text Content: {text_content[0:8000] if text_content else ""}
        Reference Topics: {topic_reference}
        Reference Resource Types: {resource_type}
        Reference Outcome Areas: {outcome_area}
        Reference Enterprise Topics: {enterprise_topic}

        ### Output
        {{
            "Relevant Topic": "...",
            "Relevant Resource Type": "...",
            "Relevant Outcome Area": [...],
            "Relevant Enterprise Topic": [...],
            "Relevant Keywords": [...]
        }}
        """

        response = az_llm_instance.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt_template,
                }
            ],
            max_tokens=16384,
            temperature=1.0,
            top_p=1.0,
            model="gpt-4o-2"
        )

        llm_output = eval(response.choices[0].message.content)
        result = {col: row[col] for col in lessdata_columns}
        result.update({
            "relevant_topic_llm": str(llm_output.get("Relevant Topic", "")),
            "relevant_resource_type_llm": str(llm_output.get("Relevant Resource Type", "")),
            "relevant_outcome_area_llm": "|".join(llm_output.get("Relevant Outcome Area", [])),
            "relevant_enterprise_topic_llm": "|".join(llm_output.get("Relevant Enterprise Topic", [])),
            "relevant_keywords_llm": "|".join(llm_output.get("Relevant Keywords", []))
        })
        results.append(result)
        continue

    if abstract_text is None or (isinstance(abstract_text, str) and abstract_text.strip() == ""):
        doc_name = getattr(row, 'DOC_NAME', '')
        doc_type_name = getattr(row, 'DOC_TYPE_NAME', '')
        maj_theme_text = getattr(row, 'MAJ_THEME_TEXT', '')
        ent_topic_text = getattr(row, 'ENT_TOPIC_TEXT', '')

        prompt_template = f"""
        ### Role
        You are an AI model that classifies document metadata into the most relevant categories strictly from provided reference lists.

        ### Instructions
        1. Always return output as a JSON object with key-value pairs.
        2. The value for 'Relevant Topic' and 'Relevant Resource Type' should be a string.
        3. The values for 'Relevant Outcome Area', 'Relevant Enterprise Topic', and 'Relevant Keywords' should be lists.
        4. If an exact match is not available, select the closest existing concept in the list.
        5. Do not change or modify the given input values while assigning the values.
        6. Do not add any additional text to the output. No markdown or HTML tags are allowed in the output.
        7. Output format must follow the structure below exactly:

        {{
            "Relevant Topic": "string",
            "Relevant Resource Type": "string",
            "Relevant Outcome Area": ["item1", "item2", "item3"],
            "Relevant Enterprise Topic": ["item1", "item2", ..., "item10"],
            "Relevant Keywords": ["item1", "item2", ..., "item10"]
        }}

        ### Input
        Document Name: {doc_name}
        Document Type Name: {doc_type_name}
        Major Theme Text: {maj_theme_text}
        Enterprise Topic Text: {ent_topic_text}
        Reference Topics: {topic_reference}
        Reference Resource Types: {resource_type}
        Reference Outcome Areas: {outcome_area}
        Reference Enterprise Topics: {enterprise_topic}

        ### Output
        {{
            "Relevant Topic": "...",
            "Relevant Resource Type": "...",
            "Relevant Outcome Area": [...],
            "Relevant Enterprise Topic": [...],
            "Relevant Keywords": [...]
        }}
        """

        response = az_llm_instance.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt_template,
                }
            ],
            max_tokens=16384,
            temperature=1.0,
            top_p=1.0,
            model="gpt-4o-2"
        )

        llm_output = eval(response.choices[0].message.content)
        result = {col: row[col] for col in lessdata_columns}
        result.update({
            "relevant_topic_llm": str(llm_output.get("Relevant Topic", "")),
            "relevant_resource_type_llm": str(llm_output.get("Relevant Resource Type", "")),
            "relevant_outcome_area_llm": "|".join(llm_output.get("Relevant Outcome Area", [])),
            "relevant_enterprise_topic_llm": "|".join(llm_output.get("Relevant Enterprise Topic", [])),
            "relevant_keywords_llm": "|".join(llm_output.get("Relevant Keywords", []))
        })
        results.append(result)
        continue

    prompt_template = f"""
    ### Role
    You are an AI model that classifies Abstract text into the most relevant categories strictly from provided reference lists.

    ### Instructions
    1. Always return output as a JSON object with key-value pairs.
    2. The value for 'Relevant Topic' and 'Relevant Resource Type' should be a string.
    3. The values for 'Relevant Outcome Area', 'Relevant Enterprise Topic', and 'Relevant Keywords' should be lists.
    4. If an exact match is not available, select the closest existing concept in the list.
    5. Ensure to not change or modify the given input values while assigning the values.
    6. Dont add any additional text to the output.No markdown or HTML tags are allowed in the output.
    7. Output format must follow the structure below exactly:

    {{
        "Relevant Topic": "string",
        "Relevant Resource Type": "string",
        "Relevant Outcome Area": ["item1", "item2", "item3"],
        "Relevant Enterprise Topic": ["item1", "item2", ..., "item10"],
        "Relevant Keywords": ["item1", "item2", ..., "item10"]
    }}

    ### Input
    Abstract Text: {abstract_text[0:8000] if abstract_text else ""}
    Reference Topics: {topic_reference}
    Reference Resource Types: {resource_type}
    Reference Outcome Areas: {outcome_area}
    Reference Enterprise Topics: {enterprise_topic}

    ### Output
    {{
        "Relevant Topic": "...",
        "Relevant Resource Type": "...",
        "Relevant Outcome Area": [...],
        "Relevant Enterprise Topic": [...],
        "Relevant Keywords": [...]
    }}
    """

    response = az_llm_instance.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt_template,
            }
        ],
        max_tokens=16384,
        temperature=1.0,
        top_p=1.0,
        model="gpt-4o"
    )

    llm_output = eval(response.choices[0].message.content)
    result = {col: row[col] for col in lessdata_columns}
    result.update({
        "relevant_topic_llm": str(llm_output.get("Relevant Topic", "")),
        "relevant_resource_type_llm": str(llm_output.get("Relevant Resource Type", "")),
        "relevant_outcome_area_llm": "|".join(llm_output.get("Relevant Outcome Area", [])),
        "relevant_enterprise_topic_llm": "|".join(llm_output.get("Relevant Enterprise Topic", [])),
        "relevant_keywords_llm": "|".join(llm_output.get("Relevant Keywords", []))
    })
    results.append(result)

schema_fields = [StructField(col, StringType(), True) for col in lessdata_columns]
schema_fields += [
    StructField("relevant_topic_llm", StringType(), True),
    StructField("relevant_resource_type_llm", StringType(), True),
    StructField("relevant_outcome_area_llm", StringType(), True),
    StructField("relevant_enterprise_topic_llm", StringType(), True),
    StructField("relevant_keywords_llm", StringType(), True)
]
schema = StructType(schema_fields)

new_df = spark.createDataFrame(results, schema=schema)
display(new_df)
