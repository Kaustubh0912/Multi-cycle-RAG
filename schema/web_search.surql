DEFINE TABLE web_search SCHEMAFULL;
DEFINE FIELD content ON web_search TYPE string;
DEFINE FIELD metadata ON web_search FLEXIBLE TYPE object;
DEFINE FIELD embedding ON web_search TYPE array<float>;
DEFINE INDEX hnsw_embedding ON web_search
FIELDS embedding
HNSW
DIMENSION 3072
DIST COSINE
TYPE F32
EFC 500
M 16;
