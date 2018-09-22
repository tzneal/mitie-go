package mitie_test

import (
	"fmt"
	"log"
	"reflect"
	"testing"

	mitie "github.com/tzneal/mitie-go"
)

func TestTokenize(t *testing.T) {
	v := mitie.Tokenize("this is a test")
	if len(v) != 4 {
		t.Errorf("expected 4, got %v", v)
	}
}

func TestTokenizeOffsets(t *testing.T) {
	s := "this is a test"
	v, o := mitie.TokenizeWithOffsets(s)
	if len(v) != 4 {
		t.Errorf("expected 4, got %v", v)
	}

	for i, v := range v {
		if s[o[i]] != v[0] {
			t.Errorf("expected %v, got %v", v[0], s[o[i]])
		}
	}
}

const txt = " John Smith was born in Huntsville, Alabama."

func TestExtractor(t *testing.T) {
	v, err := mitie.NewNamedEntityExtractor("/usr/share/mitie/english/ner_model.dat", "")
	if err != nil {
		t.Fatalf("error opening NER model: %s", err)
	}
	ents := v.Extract(txt)

	toks := mitie.Tokenize(txt)
	ents2 := v.ExtractFromTokens(toks)
	if len(ents) != 3 {
		t.Errorf("expected 3 ents, got %d", len(ents))
	}
	if !reflect.DeepEqual(ents, ents2) {
		t.Errorf("expected equal, got %v and %v", ents, ents2)
	}
	if ents[0].Name != "John Smith" {
		t.Errorf("expected 'John Smith', got %s", ents[0].Name)
	}
	if ents[0].TagStr != "PERSON" {
		t.Errorf("expected 'PERSON', got %s", ents[0].TagStr)
	}
}

func TestRelations(t *testing.T) {
	v, err := mitie.NewNamedEntityExtractor("/usr/share/mitie/english/ner_model.dat", "")
	if err != nil {
		t.Fatalf("error opening NER model: %s", err)
	}
	rd, err := mitie.NewRelationDetector("/usr/share/mitie/english/binary_relations/rel_classifier_people.person.place_of_birth.svm")
	if err != nil {
		t.Fatalf("error opening relationships model: %s", err)
	}

	ents := v.Extract(txt, rd)
	if ents[0].Relationships[0].Name != "people.person.place_of_birth" {
		t.Errorf("expected place of birth, got %v", ents[0].Relationships)
	}
}

func TestRelationsTokens(t *testing.T) {
	v, err := mitie.NewNamedEntityExtractor("/usr/share/mitie/english/ner_model.dat", "")
	if err != nil {
		t.Fatalf("error opening NER model: %s", err)
	}
	rd, err := mitie.NewRelationDetector("/usr/share/mitie/english/binary_relations/rel_classifier_people.person.place_of_birth.svm")
	if err != nil {
		t.Fatalf("error opening relationships model: %s", err)
	}

	toks := mitie.Tokenize(txt)
	ents := v.ExtractFromTokens(toks, rd)
	if ents[0].Relationships[0].Name != "people.person.place_of_birth" {
		t.Errorf("expected place of birth, got %v", ents[0].Relationships)
	}
}

func TestTraining(t *testing.T) {
	const txt = "My name is Davis King and I work for MIT."

	tng, err := mitie.NewTrainingInstance(txt)
	if err != nil {
		t.Fatalf("error creating training instance: %s", err)
	}

	if err := tng.AddEntity(3, 2, "person"); err != nil {
		t.Errorf("error adding entity: %s", err)
	}
	if err := tng.AddEntity(9, 1, "org"); err != nil {
		t.Errorf("error adding entity: %s", err)
	}

	if err := tng.AddEntity(-1, 1, "org"); err == nil {
		t.Errorf("expected error with invalid start, got none")
	}

	tng2, _ := mitie.NewTrainingInstance("The other day at work I saw Brian Smith from CMU.")
	tng2.AddEntity(7, 2, "person")
	tng2.AddEntity(10, 1, "org")

	trainer, err := mitie.NewNERTrainer("/usr/share/mitie/english/total_word_feature_extractor.dat")
	if err != nil {
		t.Fatalf("error creating trainer: %s", err)
	}

	if err := trainer.AddInstance(tng); err != nil {
		t.Errorf("error adding instance: %s", err)
	}
	if err := trainer.AddInstance(tng2); err != nil {
		t.Errorf("error adding instance: %s", err)
	}

	ner, err := trainer.Train(4, 1.0)
	if err != nil {
		t.Fatalf("error training ner: %s", err)
	}
	//ner.Save("/tmp/test.model")
	ents := ner.Extract("I met with John Becker at HBU.")
	if len(ents) != 2 {
		t.Fatalf("expected two entities, got %d", len(ents))
	}
	if ents[0].Name != "John Becker" || ents[0].TagStr != "person" {
		t.Errorf("expected name extraction, got %v", ents[0].Name)
	}
}

func ExampleNewNamedEntityExtractor_output() {
	v, err := mitie.NewNamedEntityExtractor("/usr/share/mitie/english/ner_model.dat", "")
	if err != nil {
		log.Printf("error opening NER model: %s", err)
		return
	}
	ents := v.Extract(txt)
	fmt.Println(ents)
}
