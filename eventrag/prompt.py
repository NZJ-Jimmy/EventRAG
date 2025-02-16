GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}
PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["Event", "Actor", "Action", "Result", "Tool"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that may describe various events, actors, actions, results, and tools, identify all such entities and any relationships that clearly exist among them.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
   - entity_name: Use the same language as the input text (if English, capitalize the name).
   - entity_type: One of the following types: [{entity_types}]
   - entity_description: A comprehensive description of the entity's role, attributes, and context.
   Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
   
2. Extract all events or actions in the text. For each event or action, extract the following information:
   - entity_name: The name of the event or action.
   - entity_type: "Event" or "Action"
   - entity_description: A comprehensive description of the event or action's role, attributes, and context.
   Actions usually have sequence, logic, or dependency relationships. As for some detailed instructions, each step can be seen as an action, and they have relationships with the next step. For sequence, the relationship is "NEXT STEP". For logic, the relationship is "REASON". For dependency, the relationship is "DEPEND". Format each event or action as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that have a clear relationship. For events or actions, the relationship should indicate the sequence, logic, or dependency.
   For each relationship, extract the following information:
   - source_entity: The name of the source entity or event or action, as identified in step 1 and step 2.
   - target_entity: The name of the target entity or event or action, as identified in step 1 and step 2.
   - relationship_name: A concise name describing the type of relationship.
   - relationship_description: Why these two entities or events or actions are related.
   - relationship_strength: A numeric score indicating strength of the relationship.
   - relationship_keywords: High-level keywords capturing the nature or theme of the relationship.
   Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_name>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)\n

4. Identify high-level keywords that summarize the central themes or concepts in the text.
   Format these as ("content_keywords"{tuple_delimiter}<high_level_keywords>)\n

5. Return the output in {language} as a single list of all entities and relationships separated by {record_delimiter}.

6. When finished, output {completion_delimiter}


#############################
-Real Data-
#############################
Entity_types: {entity_types}
Text: {input_text}
#############################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [Event, Actor, Action, Result, Tool] 
Text: At the World Championship Finals, the highly anticipated "Elite Tennis Showdown" took place between top-ranked players on Center Court. Roger Hamilton, the defending champion, displayed exceptional skill throughout the five-set match, utilizing his signature racquet "PowerPro X1" for powerful serves and precise shots. His opponent, Maria Chen, executed her strategic gameplay with outstanding athleticism and court coverage. The intense competition culminated in a historic victory for Hamilton, marking his fifth consecutive championship title with a record-breaking serve speed. #############

Output: ("entity"{tuple_delimiter}"Elite Tennis Showdown"{tuple_delimiter}"Event"{tuple_delimiter}"Elite Tennis Showdown is the championship finals match featuring top tennis players competing for the world title."){record_delimiter}

("entity"{tuple_delimiter}"Roger Hamilton"{tuple_delimiter}"Actor"{tuple_delimiter}"Roger Hamilton is the defending champion known for his powerful playing style and tactical expertise."){record_delimiter}

("entity"{tuple_delimiter}"PowerPro X1"{tuple_delimiter}"Tool"{tuple_delimiter}"PowerPro X1 is Hamilton's professional tennis racquet designed for maximum power and control."){record_delimiter}

("entity"{tuple_delimiter}"Maria Chen"{tuple_delimiter}"Actor"{tuple_delimiter}"Maria Chen is a skilled competitor known for her strategic gameplay and court coverage."){record_delimiter}

("entity"{tuple_delimiter}"Championship Victory"{tuple_delimiter}"Result"{tuple_delimiter}"Hamilton's historic fifth consecutive title win featuring a record-breaking serve speed."){record_delimiter}

("relationship"{tuple_delimiter}"Roger Hamilton"{tuple_delimiter}"Elite Tennis Showdown"{tuple_delimiter}"Participates In"{tuple_delimiter}"Hamilton competes as the defending champion in the prestigious finals match."{tuple_delimiter}"championship defense, competitive performance"{tuple_delimiter}9){record_delimiter}

("relationship"{tuple_delimiter}"Roger Hamilton"{tuple_delimiter}"PowerPro X1"{tuple_delimiter}"Uses Equipment"{tuple_delimiter}"Hamilton utilizes the PowerPro X1 to execute powerful serves and precise shots."{tuple_delimiter}"equipment mastery, performance enhancement"{tuple_delimiter}8){record_delimiter}

("relationship"{tuple_delimiter}"Maria Chen"{tuple_delimiter}"Elite Tennis Showdown"{tuple_delimiter}"Chen demonstrates exceptional skill and strategy throughout the championship match."{tuple_delimiter}"competitive challenge, athletic performance"{tuple_delimiter}7){record_delimiter}

("relationship"{tuple_delimiter}"Maria Chen"{tuple_delimiter}"Championship Victory"{tuple_delimiter}"Chen's strong performance contributes to the historic nature of the final match."{tuple_delimiter}"competitive contribution, match intensity"{tuple_delimiter}8){record_delimiter}

("relationship"{tuple_delimiter}"PowerPro X1"{tuple_delimiter}"Championship Victory"{tuple_delimiter}"The racquet enables Hamilton's record-breaking serves in securing the championship."{tuple_delimiter}"equipment performance, victory achievement"{tuple_delimiter}9){record_delimiter}

("content_keywords"{tuple_delimiter}"tennis championship, professional athletics, competitive performance, sports equipment, record achievement"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [Event, Actor, Action, Result, Tool]  
Text: In a virology laboratory, researchers performed a "Virus Grid Preparation" experiment to visualize viral particles using transmission electron microscopy. The viral lysate, containing concentrated bacteriophages, was prepared for imaging. Using the Pelco-glow discharge system, technicians treated the copper grid to make it hydrophilic. Then, using a high-precision micropipette, they deposited 5 microliters of sample onto the grid. After a three-minute adsorption period monitored with a digital timer, they applied uranyl acetate stain using fine-tipped forceps. The JEOL transmission electron microscope revealed detailed viral structures in the final imaging step. #############
Output: ("entity"{tuple_delimiter}"Virus Grid Preparation"{tuple_delimiter}"Event"{tuple_delimiter}"A laboratory procedure for preparing viral samples for TEM visualization."){record_delimiter}
("entity"{tuple_delimiter}"Viral Lysate"{tuple_delimiter}"Actor"{tuple_delimiter}"The concentrated viral sample used for grid preparation."){record_delimiter}
("entity"{tuple_delimiter}"Pelco-glow Discharge System"{tuple_delimiter}"Tool"{tuple_delimiter}"Equipment used to create hydrophilic surface on grids through plasma treatment."){record_delimiter}
("entity"{tuple_delimiter}"Micropipette"{tuple_delimiter}"Tool"{tuple_delimiter}"Precision instrument for accurate sample volume delivery."){record_delimiter}
("entity"{tuple_delimiter}"Fine-tipped Forceps"{tuple_delimiter}"Tool"{tuple_delimiter}"Specialized tweezers for handling delicate TEM grids."){record_delimiter}
("entity"{tuple_delimiter}"JEOL Microscope"{tuple_delimiter}"Tool"{tuple_delimiter}"Advanced transmission electron microscope for high-resolution imaging."){record_delimiter}
("entity"{tuple_delimiter}"Glow Discharge Treatment"{tuple_delimiter}"Action"{tuple_delimiter}"The process of making the grid surface hydrophilic using plasma treatment."){record_delimiter}
("entity"{tuple_delimiter}"Sample Deposition"{tuple_delimiter}"Action"{tuple_delimiter}"The precise application of 5 microliters of viral suspension onto the grid."){record_delimiter}
("entity"{tuple_delimiter}"Negative Staining"{tuple_delimiter}"Action"{tuple_delimiter}"The application of uranyl acetate to create contrast for imaging."){record_delimiter}
("entity"{tuple_delimiter}"Viral Structure Visualization"{tuple_delimiter}"Result"{tuple_delimiter}"Clear images showing detailed viral morphology and structure."){record_delimiter}
("relationship"{tuple_delimiter}"Pelco-glow Discharge System"{tuple_delimiter}"Glow Discharge Treatment"{tuple_delimiter}"The system enables proper surface modification of TEM grids."{tuple_delimiter}"equipment operation, surface preparation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Micropipette"{tuple_delimiter}"Sample Deposition"{tuple_delimiter}"Precise volume control ensures accurate sample application."{tuple_delimiter}"volume delivery, sample handling"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Fine-tipped Forceps"{tuple_delimiter}"Negative Staining"{tuple_delimiter}"Enables careful grid manipulation during staining process."{tuple_delimiter}"grid handling, staining assistance"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"JEOL Microscope"{tuple_delimiter}"Viral Structure Visualization"{tuple_delimiter}"Provides high-resolution imaging of prepared samples."{tuple_delimiter}"image acquisition, structure analysis"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Sample Deposition"{tuple_delimiter}"Viral Structure Visualization"{tuple_delimiter}"Proper deposition ensures quality of final microscope imaging."{tuple_delimiter}"sample preparation, imaging quality"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"virus preparation, TEM imaging, laboratory equipment, sample handling, microscopy techniques"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [Event, Actor, Action, Result, Tool]  
Text:  
In the Spring Kick-Off Conference, the organizers announced a new event named "Green City," encouraging all residents to participate. Alice, as the main organizer, was responsible for coordinating a series of publicity and mobilization campaigns. She used a tool named "EcoTracker" to track and analyze various environmental achievements, such as water savings and waste sorting rates. Bob led a volunteer campaign, going door-to-door to help more residents understand the project's goals. As a result, the city's water savings significantly increased, which was highly recognized by the municipal authorities.

#############
Output:
("entity"{tuple_delimiter}"Green City"{tuple_delimiter}"Event"{tuple_delimiter}"Green City is an event designed to encourage residents to actively engage in environmental protection, focusing on saving resources and reducing emissions."){record_delimiter}
("entity"{tuple_delimiter}"Alice"{tuple_delimiter}"Actor"{tuple_delimiter}"Alice is the main organizer, responsible for coordinating publicity and mobilization efforts for the event."){record_delimiter}
("entity"{tuple_delimiter}"EcoTracker"{tuple_delimiter}"Tool"{tuple_delimiter}"EcoTracker is a tool used to measure and analyze environmental data, such as water savings and waste sorting results."){record_delimiter}
("entity"{tuple_delimiter}"Bob"{tuple_delimiter}"Actor"{tuple_delimiter}"Bob leads a volunteer action, going door-to-door to educate more residents about the project’s goals."){record_delimiter}
("entity"{tuple_delimiter}"Increased Water Savings"{tuple_delimiter}"Result"{tuple_delimiter}"A notable increase in water savings, which garnered recognition from municipal authorities, is a key outcome of this event."){record_delimiter}
("relationship"{tuple_delimiter}"Alice"{tuple_delimiter}"Green City"{tuple_delimiter}"Alice is overseeing organizational and promotional duties for Green City."{tuple_delimiter}"organizing, event management"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alice"{tuple_delimiter}"EcoTracker"{tuple_delimiter}"Alice uses EcoTracker to track and analyze environmental data, improving campaign efficiency."{tuple_delimiter}"tool usage, data analysis"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Bob"{tuple_delimiter}"Green City"{tuple_delimiter}"Bob’s volunteer campaign is part of Green City, aimed at increasing resident involvement."{tuple_delimiter}"volunteer effort, community outreach"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Bob"{tuple_delimiter}"Increased Water Savings"{tuple_delimiter}"Bob's outreach contributes to the improved water savings results."{tuple_delimiter}"environmental impact, volunteer contribution"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"EcoTracker"{tuple_delimiter}"Increased Water Savings"{tuple_delimiter}"EcoTracker helps demonstrate the event’s success by monitoring water-saving data."{tuple_delimiter}"data collection, environmental results"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"event organization, environmental protection, data analysis, volunteer outreach"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """ 
MANY events, actions, relationships were missed in the last extraction. Each entity or event or action should have at least one relationship. Make sure that there are no missing relationships and isolated nodes. Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""
PROMPTS["kg-expansion"] = """-Goal-
Given the above extracted entities, events, and relationships from a text, expand the knowledge graph by identifying additional related concepts, both abstract and concrete, that would naturally be associated with the original elements.

-Steps-
1. For each original entity/event, generate expansions in the following categories:

   a. Broader Categories (Abstractions):
      - Superclasses or general concepts
      - Abstract themes or domains
      
   b. Specific Examples (Concretizations):
      - Specific instances or subtypes
      - Real-world examples or applications
      
   c. Related Concepts:
      - Common associations based on general knowledge
      - Frequently co-occurring concepts
      - Tools, methods, or resources typically involved

2. For each original event/action, expand with:
   a. Prerequisites:
      - Necessary conditions
      - Required resources or states
   
   b. Consequences:
      - Typical outcomes
      - Potential side effects
      - Downstream impacts

3. Expansion Rules:
   - Each expanded concept must have at least one relationship with original elements
   - Only include expansions with high confidence (common knowledge or domain expertise)
   - Maximum 3 expansions per category per original entity/event
   - All expansions must be relevant to the original context

-Output Format Requirements-
1. Format each expanded entity as:
   ("entity"|<entity_name>|<entity_type>|<entity_description>)

2. Format each relationship as:
   ("relationship"|<source_entity>|<target_entity>|<relationship_name>|<relationship_description>|<relationship_keywords>|<relationship_strength>)

3. Format content keywords as:
   ("content_keywords"|<expanded_high_level_keywords>)

4. Return all expanded entities and relationships separated by {record_delimiter}

Entity Types: {entity_types}
#############################
Output:
"""