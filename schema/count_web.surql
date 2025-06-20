DEFINE FUNCTION OVERWRITE fn::count_web() -> int {
	RETURN count((SELECT * FROM web_search));
} PERMISSIONS FULL