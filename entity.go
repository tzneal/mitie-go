package mitie

// Entity is an entity that has been extracted by NamedEntityExtractor
type Entity struct {
	Name   string  // name of the entity
	Pos    int     // position within the token list
	Len    int     // length of the entity in tokens (e.g. 2 for "John Doe", 1 for "John")
	Tag    int     // numeric tag of the entity
	TagStr string  // string version of the nummeric tag (e.g. "PERSON")
	Score  float64 // match score

	Relationships []Relationship
}

// Relationship is a binary relationship found by a RelationDetector
type Relationship struct {
	Name  string
	Other Entity
	Score float64
}
