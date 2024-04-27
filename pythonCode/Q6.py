expanded_query_list = []
def query_expansion(query, knowledge_graph):
    query_text = str(query[1] if isinstance(query, tuple) else query)
    query_entities = extract_entities(query_text)
    expanded_query_entities = set(query_entities)

    if not query_entities:
        print(f"Warning: No entities found in the query '{query_text}'")
        return query_text

    for entity in query_entities:
        if entity not in knowledge_graph:
            print(f"Warning: Entity '{entity}' not found in the knowledge graph")
        else:
            for relation, related_entity in knowledge_graph[entity]:
                expanded_query_entities.add(related_entity)

    if len(expanded_query_entities) == len(query_entities):
        print(f"Warning: No new entities added to the query '{query_text}'")

    expanded_query = " ".join(expanded_query_entities)
    expanded_query_list.append(expanded_query)
    return expanded_query



# Experiment 6: Query Expansion using Knowledge Graph
for expanded_query in expanded_query_list:
    query_text = expanded_query[1] if isinstance(query, tuple) else query
    expanded_query = query_expansion(query_text, knowledge_graph)
    print(f"Original Query: {query}")
    print(f"Expanded Query: {expanded_query}")
    retrieve_and_print_documents(expanded_query_list, documents, num_queries=10)
    
