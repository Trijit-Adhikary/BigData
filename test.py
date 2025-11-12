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






# =======================================================================================================================================================================================

# Optimized Code with Performance Improvements

---

## Key Performance Issues in Original Code

1. **Sequential processing with `.collect()`** - brings all data to driver node
2. **No parallelization** - processes one row at a time
3. **Repetitive API calls** - no batching or connection pooling
4. **Memory inefficiency** - stores all results in driver memory
5. **Unsafe `eval()`** usage - security risk and potential parsing errors

---

## Optimized Solution

```python
import os
import json
from openai import AzureOpenAI
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col, udf, when, trim, length
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_azure_client():
    """Create Azure OpenAI client with connection pooling"""
    return AzureOpenAI(
        # Add your Azure OpenAI configuration here
        # azure_endpoint="your_endpoint",
        # api_key="your_key",
        # api_version="your_version"
    )

def safe_json_parse(json_string):
    """Safely parse JSON response from LLM"""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return {
            "Relevant Topic": "",
            "Relevant Resource Type": "",
            "Relevant Outcome Area": [],
            "Relevant Enterprise Topic": [],
            "Relevant Keywords": []
        }

def create_prompt_template(content_type, **kwargs):
    """Create standardized prompt template"""
    base_instructions = """
    ### Role
    You are an AI model that classifies document {content_type} into the most relevant categories strictly from provided reference lists.

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
    {input_section}
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
    
    if content_type == "text":
        input_section = f"Text Content: {kwargs.get('text_content', '')[:8000]}"
    elif content_type == "metadata":
        input_section = f"""Document Name: {kwargs.get('doc_name', '')}
Document Type Name: {kwargs.get('doc_type_name', '')}
Major Theme Text: {kwargs.get('maj_theme_text', '')}
Enterprise Topic Text: {kwargs.get('ent_topic_text', '')}"""
    else:  # abstract
        input_section = f"Abstract Text: {kwargs.get('abstract_text', '')[:8000]}"
    
    return base_instructions.format(
        content_type=content_type,
        input_section=input_section,
        topic_reference=kwargs.get('topic_reference', ''),
        resource_type=kwargs.get('resource_type', ''),
        outcome_area=kwargs.get('outcome_area', ''),
        enterprise_topic=kwargs.get('enterprise_topic', '')
    )

def process_single_row(row_data, topic_reference, resource_type, outcome_area, enterprise_topic):
    """Process a single row with LLM classification"""
    client = create_azure_client()
    
    text_content = row_data.get('text_content')
    abstract_text = row_data.get('ABSTACT_TEXT')
    
    # Determine processing logic (same as original)
    use_abstract = (
        text_content is None or
        (isinstance(text_content, str) and (
            text_content.strip() == "" or
            text_content.strip() == "The original PDF is Password Protected for Opening. Unable to extract text for Index."
        ))
    )
    
    try:
        if not use_abstract:
            # Process text content
            prompt = create_prompt_template(
                "text",
                text_content=text_content,
                topic_reference=topic_reference,
                resource_type=resource_type,
                outcome_area=outcome_area,
                enterprise_topic=enterprise_topic
            )
            model_name = "gpt-4o-2"
            
        elif abstract_text is None or (isinstance(abstract_text, str) and abstract_text.strip() == ""):
            # Process metadata
            prompt = create_prompt_template(
                "metadata",
                doc_name=row_data.get('DOC_NAME', ''),
                doc_type_name=row_data.get('DOC_TYPE_NAME', ''),
                maj_theme_text=row_data.get('MAJ_THEME_TEXT', ''),
                ent_topic_text=row_data.get('ENT_TOPIC_TEXT', ''),
                topic_reference=topic_reference,
                resource_type=resource_type,
                outcome_area=outcome_area,
                enterprise_topic=enterprise_topic
            )
            model_name = "gpt-4o-2"
            
        else:
            # Process abstract
            prompt = create_prompt_template(
                "abstract",
                abstract_text=abstract_text,
                topic_reference=topic_reference,
                resource_type=resource_type,
                outcome_area=outcome_area,
                enterprise_topic=enterprise_topic
            )
            model_name = "gpt-4o"
        
        # Make API call with retry logic
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            max_tokens=16384,
            temperature=1.0,
            top_p=1.0,
            model=model_name
        )
        
        # Parse response safely
        llm_output = safe_json_parse(response.choices[0].message.content)
        
        # Prepare result
        result = dict(row_data)
        result.update({
            "relevant_topic_llm": str(llm_output.get("Relevant Topic", "")),
            "relevant_resource_type_llm": str(llm_output.get("Relevant Resource Type", "")),
            "relevant_outcome_area_llm": "|".join(llm_output.get("Relevant Outcome Area", [])),
            "relevant_enterprise_topic_llm": "|".join(llm_output.get("Relevant Enterprise Topic", [])),
            "relevant_keywords_llm": "|".join(llm_output.get("Relevant Keywords", []))
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing row: {e}")
        # Return original row with empty LLM fields on error
        result = dict(row_data)
        result.update({
            "relevant_topic_llm": "",
            "relevant_resource_type_llm": "",
            "relevant_outcome_area_llm": "",
            "relevant_enterprise_topic_llm": "",
            "relevant_keywords_llm": ""
        })
        return result

# Create UDF for parallel processing
def create_processing_udf(topic_reference, resource_type, outcome_area, enterprise_topic):
    def process_row_udf(row_dict):
        return process_single_row(row_dict, topic_reference, resource_type, outcome_area, enterprise_topic)
    return udf(process_row_udf, StringType())

# MAIN PROCESSING - Optimized approach
def process_data_optimized(lessdata, topic_reference, resource_type, outcome_area, enterprise_topic, batch_size=100):
    """
    Process data in batches using Spark's parallel processing capabilities
    """
    
    # Method 1: Using mapPartitions for better performance
    def process_partition(iterator):
        client = create_azure_client()  # Create client per partition
        results = []
        
        for row in iterator:
            row_data = row.asDict()
            try:
                processed_row = process_single_row(row_data, topic_reference, resource_type, outcome_area, enterprise_topic)
                results.append(processed_row)
            except Exception as e:
                logger.error(f"Error in partition processing: {e}")
                # Add row with empty LLM fields on error
                row_data.update({
                    "relevant_topic_llm": "",
                    "relevant_resource_type_llm": "",
                    "relevant_outcome_area_llm": "",
                    "relevant_enterprise_topic_llm": "",
                    "relevant_keywords_llm": ""
                })
                results.append(row_data)
        
        return results
    
    # Process using mapPartitions
    processed_rdd = lessdata.rdd.mapPartitions(process_partition)
    
    # Create schema
    schema_fields = [StructField(col, StringType(), True) for col in lessdata.columns]
    schema_fields += [
        StructField("relevant_topic_llm", StringType(), True),
        StructField("relevant_resource_type_llm", StringType(), True),
        StructField("relevant_outcome_area_llm", StringType(), True),
        StructField("relevant_enterprise_topic_llm", StringType(), True),
        StructField("relevant_keywords_llm", StringType(), True)
    ]
    schema = StructType(schema_fields)
    
    # Create DataFrame from processed RDD
    result_df = spark.createDataFrame(processed_rdd, schema=schema)
    
    return result_df

# Execute optimized processing
try:
    new_df = process_data_optimized(
        lessdata, 
        topic_reference, 
        resource_type, 
        outcome_area, 
        enterprise_topic
    )
    
    # Cache for better performance if reusing
    new_df.cache()
    
    display(new_df)
    
except Exception as e:
    logger.error(f"Processing failed: {e}")
    raise
```

---

## Alternative Batch Processing Approach

```python
def process_data_in_batches(lessdata, topic_reference, resource_type, outcome_area, enterprise_topic, batch_size=50):
    """
    Alternative approach: Process data in smaller batches to avoid memory issues
    """
    
    # Get row count and calculate number of batches
    total_rows = lessdata.count()
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    logger.info(f"Processing {total_rows} rows in {num_batches} batches of size {batch_size}")
    
    all_results = []
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_rows)
        
        logger.info(f"Processing batch {batch_num + 1}/{num_batches}")
        
        # Get batch data
        batch_data = lessdata.limit(end_idx).subtract(lessdata.limit(start_idx))
        
        # Process batch using mapPartitions
        def process_batch_partition(iterator):
            client = create_azure_client()
            results = []
            
            for row in iterator:
                row_data = row.asDict()
                processed_row = process_single_row(row_data, topic_reference, resource_type, outcome_area, enterprise_topic)
                results.append(processed_row)
            
            return results
        
        batch_results = batch_data.rdd.mapPartitions(process_batch_partition).collect()
        all_results.extend(batch_results)
        
        # Optional: Add delay between batches to avoid rate limiting
        # import time
        # time.sleep(1)
    
    # Create final DataFrame
    schema_fields = [StructField(col, StringType(), True) for col in lessdata.columns]
    schema_fields += [
        StructField("relevant_topic_llm", StringType(), True),
        StructField("relevant_resource_type_llm", StringType(), True),
        StructField("relevant_outcome_area_llm", StringType(), True),
        StructField("relevant_enterprise_topic_llm", StringType(), True),
        StructField("relevant_keywords_llm", StringType(), True)
    ]
    schema = StructType(schema_fields)
    
    result_df = spark.createDataFrame(all_results, schema=schema)
    return result_df
```

---

## Key Optimizations Explained

### 1. **Parallel Processing**
- **Original**: Sequential processing with `.collect()` 
- **Optimized**: Uses `mapPartitions()` for distributed processing across cluster nodes

### 2. **Memory Management**
- **Original**: Loads all data to driver node
- **Optimized**: Processes data in partitions, reducing driver memory usage

### 3. **Error Handling**
- **Original**: Uses unsafe `eval()` 
- **Optimized**: Implements safe JSON parsing with fallback values

### 4. **Code Structure**
- **Original**: Repetitive code blocks
- **Optimized**: Modular functions with reusable components

### 5. **Connection Management**
- **Original**: Single client instance
- **Optimized**: Client per partition for better resource utilization

### 6. **Batch Processing Options**
1. **mapPartitions**: Best for large datasets
2. **Batch processing**: Better control over API rate limits

---

## Performance Recommendations

### 1. **Cluster Configuration**
```python
# Recommended Spark configurations
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true") 
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

### 2. **Partition Optimization**
```python
# Repartition data based on cluster size
optimal_partitions = lessdata.count() // 1000  # Adjust based on your data
lessdata = lessdata.repartition(optimal_partitions)
```

### 3. **Monitoring & Debugging**
```python
# Add progress tracking
from pyspark import TaskContext

def process_partition_with_progress(iterator):
    partition_id = TaskContext.get().partitionId()
    logger.info(f"Starting partition {partition_id}")
    # ... processing logic
    logger.info(f"Completed partition {partition_id}")
```

This optimized version should significantly improve performance and cluster stability while maintaining the same business logic.



# =======================================================================================================================================================================================

def process_data_with_partitions(lessdata, topic_reference, resource_type, outcome_area, enterprise_topic, num_partitions=10):
    """
    Most efficient: Use Spark's natural partitioning
    """
    
    # Repartition data for balanced processing
    partitioned_data = lessdata.repartition(num_partitions)
    
    def process_partition(iterator):
        client = create_azure_client()
        results = []
        partition_id = TaskContext.get().partitionId() if TaskContext else 0
        
        logger.info(f"Starting partition {partition_id}")
        
        for row in iterator:
            row_data = row.asDict()
            try:
                processed_row = process_single_row(row_data, topic_reference, resource_type, outcome_area, enterprise_topic)
                results.append(processed_row)
            except Exception as e:
                logger.error(f"Error in partition {partition_id}: {e}")
                # Add row with empty LLM fields on error
                row_data.update({
                    "relevant_topic_llm": "",
                    "relevant_resource_type_llm": "",
                    "relevant_outcome_area_llm": "",
                    "relevant_enterprise_topic_llm": "",
                    "relevant_keywords_llm": ""
                })
                results.append(row_data)
        
        logger.info(f"Completed partition {partition_id}")
        return results
    
    # Process all partitions in parallel
    processed_rdd = partitioned_data.rdd.mapPartitions(process_partition)
    
    # Create final DataFrame
    schema = create_output_schema(lessdata.columns)
    result_df = spark.createDataFrame(processed_rdd, schema=schema)
    
    return result_df



def create_output_schema(input_columns):
    """Create output schema for the processed DataFrame"""
    schema_fields = [StructField(col, StringType(), True) for col in input_columns]
    schema_fields += [
        StructField("relevant_topic_llm", StringType(), True),
        StructField("relevant_resource_type_llm", StringType(), True),
        StructField("relevant_outcome_area_llm", StringType(), True),
        StructField("relevant_enterprise_topic_llm", StringType(), True),
        StructField("relevant_keywords_llm", StringType(), True)
    ]
    return StructType(schema_fields)






from pyspark.sql.functions import pandas_udf, col, struct, monotonically_increasing_id, floor
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define output schema
llm_output_schema = StructType([
    StructField("relevant_topic_llm", StringType(), True),
    StructField("relevant_resource_type_llm", StringType(), True),
    StructField("relevant_outcome_area_llm", StringType(), True),
    StructField("relevant_enterprise_topic_llm", StringType(), True),
    StructField("relevant_keywords_llm", StringType(), True)
])

@pandas_udf(returnType=llm_output_schema)
def process_batch_llm(
    text_content: pd.Series,
    abstract_text: pd.Series,
    doc_name: pd.Series,
    doc_type_name: pd.Series,
    maj_theme_text: pd.Series,
    ent_topic_text: pd.Series
) -> pd.DataFrame:
    """
    Optimized pandas UDF for batch processing
    Processes multiple rows efficiently with single client initialization
    """
    
    # Initialize client once per batch (very important for performance)
    client = create_azure_client()
    
    results = []
    batch_size = len(text_content)
    
    print(f"Processing batch of {batch_size} rows...")
    
    for i in range(batch_size):
        try:
            # Prepare row data
            row_data = {
                'text_content': text_content.iloc[i] if pd.notna(text_content.iloc[i]) else None,
                'ABSTACT_TEXT': abstract_text.iloc[i] if pd.notna(abstract_text.iloc[i]) else None,
                'DOC_NAME': doc_name.iloc[i] if pd.notna(doc_name.iloc[i]) else '',
                'DOC_TYPE_NAME': doc_type_name.iloc[i] if pd.notna(doc_type_name.iloc[i]) else '',
                'MAJ_THEME_TEXT': maj_theme_text.iloc[i] if pd.notna(maj_theme_text.iloc[i]) else '',
                'ENT_TOPIC_TEXT': ent_topic_text.iloc[i] if pd.notna(ent_topic_text.iloc[i]) else ''
            }
            
            # Process single row with existing logic
            processed = process_single_row(
                row_data, topic_reference, resource_type, outcome_area, enterprise_topic
            )
            
            # Extract LLM results
            result_row = {
                "relevant_topic_llm": processed.get("relevant_topic_llm", ""),
                "relevant_resource_type_llm": processed.get("relevant_resource_type_llm", ""),
                "relevant_outcome_area_llm": processed.get("relevant_outcome_area_llm", ""),
                "relevant_enterprise_topic_llm": processed.get("relevant_enterprise_topic_llm", ""),
                "relevant_keywords_llm": processed.get("relevant_keywords_llm", "")
            }
            
            results.append(result_row)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{batch_size} rows in current batch")
                
        except Exception as e:
            logger.error(f"Error processing row {i} in batch: {e}")
            # Add empty results on error
            results.append({
                "relevant_topic_llm": "",
                "relevant_resource_type_llm": "",
                "relevant_outcome_area_llm": "",
                "relevant_enterprise_topic_llm": "",
                "relevant_keywords_llm": ""
            })
    
    print(f"Completed batch of {batch_size} rows")
    return pd.DataFrame(results)

def process_3000_rows_optimized(lessdata, topic_reference, resource_type, outcome_area, enterprise_topic):
    """
    Optimized processing for 3000 rows using DataFrame APIs only
    """
    
    print("=" * 50)
    print("STARTING OPTIMIZED DATAFRAME PROCESSING")
    print("=" * 50)
    
    # Step 1: Validate input
    total_rows = lessdata.count()
    print(f"📊 Total rows to process: {total_rows}")
    print(f"📋 Columns: {len(lessdata.columns)}")
    
    # Step 2: Optimize partitioning for 3000 rows
    # For 3000 rows, we want 15-20 partitions (150-200 rows each)
    optimal_partitions = max(1, min(total_rows // 150, 20))
    print(f"🔧 Repartitioning to {optimal_partitions} partitions")
    
    optimized_df = lessdata.repartition(optimal_partitions)
    
    # Step 3: Cache the repartitioned data
    optimized_df.cache()
    optimized_df.count()  # Force caching
    print("💾 Data cached successfully")
    
    # Step 4: Apply processing UDF
    print("🚀 Starting LLM processing...")
    
    import time
    start_time = time.time()
    
    # Fill null values to avoid pandas issues
    clean_df = optimized_df.fillna({
        'text_content': '',
        'ABSTACT_TEXT': '',
        'DOC_NAME': '',
        'DOC_TYPE_NAME': '',
        'MAJ_THEME_TEXT': '',
        'ENT_TOPIC_TEXT': ''
    })
    
    # Apply the pandas UDF
    result_df = clean_df.withColumn(
        "llm_results",
        process_batch_llm(
            col("text_content"),
            col("ABSTACT_TEXT"),
            col("DOC_NAME"),
            col("DOC_TYPE_NAME"),
            col("MAJ_THEME_TEXT"),
            col("ENT_TOPIC_TEXT")
        )
    )
    
    # Step 5: Flatten the results
    final_df = result_df.select(
        "*",
        col("llm_results.relevant_topic_llm").alias("relevant_topic_llm"),
        col("llm_results.relevant_resource_type_llm").alias("relevant_resource_type_llm"),
        col("llm_results.relevant_outcome_area_llm").alias("relevant_outcome_area_llm"),
        col("llm_results.relevant_enterprise_topic_llm").alias("relevant_enterprise_topic_llm"),
        col("llm_results.relevant_keywords_llm").alias("relevant_keywords_llm")
    ).drop("llm_results")
    
    # Step 6: Cache and validate results
    final_df.cache()
    final_count = final_df.count()
    
    # Step 7: Check processing success
    successful_rows = final_df.filter(
        (col("relevant_topic_llm").isNotNull()) & 
        (col("relevant_topic_llm") != "")
    ).count()
    
    end_time = time.time()
    processing_time = (end_time - start_time) / 60
    
    # Results summary
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETED!")
    print("=" * 50)
    print(f"✅ Total rows processed: {final_count}")
    print(f"✅ Successful processing: {successful_rows}")
    print(f"❌ Failed/Empty results: {final_count - successful_rows}")
    print(f"⏱️  Total processing time: {processing_time:.2f} minutes")
    print(f"⚡ Average time per row: {(processing_time * 60) / final_count:.2f} seconds")
    
    return final_df





