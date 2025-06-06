DEFINE FUNCTION OVERWRITE fn::hybrid_search($query_text: string, $query_embedding: array<float>, $k: int) {
	LET $limit = $k ?? 5;
	
	SELECT 
	    id, 
	    content, 
	    metadata, 
	    (vector::similarity::cosine(embedding, $query_embedding) * 0.7f) + (search::score($query_text) * 0.3f) 
	
	    AS combined_score 
	
	FROM documents 
	    WHERE embedding <|50,COSINE|> $query_embedding 
	    OR content @1@ $query_text 
	    ORDER BY combined_score DESC
	LIMIT $limit;
} PERMISSIONS FULL