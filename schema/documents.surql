-- Define documents table
DEFINE TABLE documents SCHEMAFULL;
DEFINE FIELD content ON documents TYPE string;
DEFINE FIELD metadata ON documents FLEXIBLE TYPE object;
DEFINE FIELD embedding ON documents TYPE array<float>;
DEFINE INDEX hnsw_embedding ON documents
FIELDS embedding
HNSW
DIMENSION 3072
DIST COSINE
TYPE F32
EFC 500
M 16;
