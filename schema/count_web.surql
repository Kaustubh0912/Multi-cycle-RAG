DEFINE FUNCTION OVERWRITE fn::count_web() -> int {
	RETURN count((SELECT * FROM documents));
} PERMISSIONS FULL