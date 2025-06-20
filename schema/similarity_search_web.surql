DEFINE FUNCTION OVERWRITE fn::similarity_search_web($query_embedding: array<float>, $k: int) -> any {
	LET $limit = $k ?? 5;
	RETURN (SELECT id, content, metadata, vector::similarity::cosine(embedding, $query_embedding) AS score FROM web_search WHERE embedding <|300,COSINE|> $query_embedding ORDER BY score DESC
	 LIMIT $limit);
} PERMISSIONS FULL