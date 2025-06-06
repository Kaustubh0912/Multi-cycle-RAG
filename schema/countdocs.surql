DEFINE FUNCTION OVERWRITE fn::countdocs() -> int {
	RETURN count((SELECT * FROM documents));
} PERMISSIONS FULL