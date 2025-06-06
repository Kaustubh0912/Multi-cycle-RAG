DEFINE FUNCTION OVERWRITE fn::deldocs($confirm: string) -> string {
	LET $word = 'CONFIRM';
	IF $confirm == $word
		{
	
			DELETE documents;
	
			RETURN 'DELETED';
		
	}
	ELSE
		{
			RETURN 'NOT DELETED';
		}
	;
} PERMISSIONS FULL