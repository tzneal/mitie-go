package mitie

/*
#cgo LDFLAGS: -lmitie

#include <stdlib.h>

#include "mitie.h"
char **newCharArray(int len) {
   return calloc(len,sizeof(char*));
}
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"unsafe"
)

// NewNamedEntityExtractor contstruct a new NamedEntityExtractor given
// a model and optional feature extractor.
func NewNamedEntityExtractor(filename, feFilename string) (*NamedEntityExtractor, error) {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	isPure := C.mitie_check_ner_pure_model(cs) == 0
	if isPure {
		// pure model, but no feature extractor file
		if feFilename == "" {
			ex := C.mitie_load_named_entity_extractor_pure_model_without_feature_extractor(cs)
			if ex == nil {
				return nil, errors.New("unable to load extractor")
			}
			ner := &NamedEntityExtractor{ex}
			runtime.SetFinalizer(ner, freeNamedEntityExtractor)
			return ner, nil
		}
		// pure model with feature extractor
		feCs := C.CString(filename)
		defer C.free(unsafe.Pointer(feCs))
		ex := C.mitie_load_named_entity_extractor_pure_model(cs, feCs)
		if ex == nil {
			return nil, errors.New("unable to load extractor")
		}
		ner := &NamedEntityExtractor{ex}
		runtime.SetFinalizer(ner, freeNamedEntityExtractor)
		return ner, nil
	}

	ex := C.mitie_load_named_entity_extractor(cs)
	if ex == nil {
		return nil, errors.New("unable to load extractor")
	}
	ner := &NamedEntityExtractor{ex}
	runtime.SetFinalizer(ner, freeNamedEntityExtractor)
	return ner, nil
}

func freeNamedEntityExtractor(n *NamedEntityExtractor) {
	C.mitie_free(unsafe.Pointer(n.x))
}

// NamedEntityExtractor is a MITIE NER that uses a trained model to
// extract named entities from plain text.
type NamedEntityExtractor struct {
	x *C.mitie_named_entity_extractor
}

// Save saves a NamedEntityExtractor model (usually a new model after
// training) to a file on disk.
func (n NamedEntityExtractor) Save(filename string) error {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))

	c := C.mitie_save_named_entity_extractor(cs, n.x)
	if c != 0 {
		return fmt.Errorf("error saving NER: %d", c)
	}
	return nil
}

// PossibleTags returns the list of possible named entity tags for
// this NamedEntityExtractor.
func (n NamedEntityExtractor) PossibleTags() []string {
	numTags := C.mitie_get_num_possible_ner_tags(n.x)
	tags := []string{}
	for i := C.ulong(0); i < numTags; i++ {
		// don't need to free these, they're freed when the ner is freed
		tags = append(tags, C.GoString(C.mitie_get_named_entity_tagstr(n.x, i)))
	}
	return tags
}

// ExtractFromTokens extracts entities from a list of tokens.
func (n NamedEntityExtractor) ExtractFromTokens(toks []string, rds ...*RelationDetector) []Entity {

	// create a char** with our tokens
	rawArr := C.newCharArray(C.int(len(toks) + 1))
	arr := (*[1 << 30]*C.char)(unsafe.Pointer(rawArr))[:len(toks)]
	defer C.free(unsafe.Pointer(rawArr))
	for i, t := range toks {
		arr[i] = C.CString(t)
	}

	// extract entities
	dets := C.mitie_extract_entities(n.x, rawArr)
	if dets == nil {
		return nil
	}
	defer C.mitie_free(unsafe.Pointer(dets))

	// parse the detections
	numDets := int(C.mitie_ner_get_num_detections(dets))
	ents := []Entity{}
	tags := n.PossibleTags()
	for i := 0; i < numDets; i++ {
		tag := int(C.mitie_ner_get_detection_tag(dets, C.ulong(i)))
		pos := int(C.mitie_ner_get_detection_position(dets, C.ulong(i)))
		len := int(C.mitie_ner_get_detection_length(dets, C.ulong(i)))

		ents = append(ents, Entity{
			Pos:    pos,
			Len:    len,
			Tag:    tag,
			TagStr: tags[tag],
			Score:  float64(C.mitie_ner_get_detection_score(dets, C.ulong(i))),
			Name:   strings.Join(toks[pos:pos+len], " "),
		})
	}

	n.applyRds(rds, rawArr, ents)

	for _, v := range arr {
		C.free(unsafe.Pointer(v))
	}
	return ents
}

// Extract extracts entities from a list of tokens and is equivalent
// to calling ExtractFromTokens(Tokenize(s))
func (n NamedEntityExtractor) Extract(s string, rds ...*RelationDetector) []Entity {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))

	// tokenize
	toks := C.mitie_tokenize(cs)
	if toks == nil {
		return nil
	}
	defer C.mitie_free(unsafe.Pointer(toks))

	// extract entities
	dets := C.mitie_extract_entities(n.x, toks)
	if dets == nil {
		// no entities found
		return nil
	}
	defer C.mitie_free(unsafe.Pointer(dets))

	// parse the detections
	numDets := int(C.mitie_ner_get_num_detections(dets))
	tokens := (*[1 << 30]*C.char)(unsafe.Pointer(toks))
	ents := []Entity{}
	tags := n.PossibleTags()
	for i := 0; i < numDets; i++ {
		tag := C.mitie_ner_get_detection_tag(dets, C.ulong(i))
		pos := C.mitie_ner_get_detection_position(dets, C.ulong(i))
		len := C.mitie_ner_get_detection_length(dets, C.ulong(i))

		name := []string{}
		for _, t := range tokens[pos : pos+len] {
			name = append(name, C.GoString(t))
		}

		ents = append(ents, Entity{
			Pos:    int(pos),
			Len:    int(len),
			Tag:    int(tag),
			TagStr: tags[int(tag)],
			Score:  float64(C.mitie_ner_get_detection_score(dets, C.ulong(i))),
			Name:   strings.Join(name, " "),
		})

	}
	n.applyRds(rds, toks, ents)
	return ents
}

func (n NamedEntityExtractor) applyRds(rds []*RelationDetector, toks **C.char, ents []Entity) {
	for i := 0; i+1 < len(ents); i++ {
		rel1 := C.mitie_extract_binary_relation(n.x, toks, C.ulong(ents[i].Pos), C.ulong(ents[i].Len),
			C.ulong(ents[i+1].Pos), C.ulong(ents[i+1].Len))
		rel2 := C.mitie_extract_binary_relation(n.x, toks, C.ulong(ents[i+1].Pos), C.ulong(ents[i+1].Len),
			C.ulong(ents[i].Pos), C.ulong(ents[i].Len))

		var score C.double
		for _, rd := range rds {
			rdName := rd.Name()
			if C.mitie_classify_binary_relation(rd.x, rel1, &score) == 0 {
				if score > 0 {
					ents[i].Relationships = append(ents[i].Relationships,
						Relationship{Name: rdName, Other: ents[i+1], Score: float64(score)})
				}
			}
			if C.mitie_classify_binary_relation(rd.x, rel2, &score) == 0 {
				if score > 0 {
					ents[i+1].Relationships = append(ents[i+1].Relationships,
						Relationship{Name: rdName, Other: ents[i], Score: float64(score)})
				}
			}
		}
		C.mitie_free(unsafe.Pointer(rel1))
		C.mitie_free(unsafe.Pointer(rel2))
	}
}
