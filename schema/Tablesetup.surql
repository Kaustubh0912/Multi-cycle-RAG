-- Define documents table
DEFINE TABLE documents SCHEMAFULL;
DEFINE FIELD content ON documents TYPE string;
DEFINE FIELD metadata ON documents TYPE object;
DEFINE FIELD embedding ON documents TYPE array<float>;
DEFINE INDEX hnsw_embedding ON documents FIELDS embedding HNSW DIMENSION 384;

-- Define embedding model reference
DEFINE TABLE embedding_model SCHEMAFULL;
DEFINE FIELD model_name ON embedding_model TYPE string;
DEFINE FIELD dimensions ON embedding_model TYPE int;


DEFINE FUNCTION fn::similarity_search($query_embedding: array<float>, $k: int) {
    LET $limit = $k ?? 5;
    SELECT id, content, metadata,
           vector::similarity::cosine(embedding, $query_embedding) AS score
    FROM documents
    WHERE embedding <|50,COSINE|> $query_embedding
    ORDER BY score DESC
    LIMIT $limit;
};
DEFINE FUNCTION fn::hybrid_search($query_text: string, $query_embedding: array<float>, $k: int) {
    LET $limit = $k ?? 5;
    SELECT id, content, metadata,
           (vector::similarity::cosine(embedding, $query_embedding) * 0.7) +
           (search::score($query_text) * 0.3) AS combined_score
    FROM documents
    WHERE embedding <|50,COSINE|> $query_embedding
       OR content @1@ $query_text
    ORDER BY combined_score DESC
    LIMIT $limit;
};
