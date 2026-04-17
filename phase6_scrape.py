"""
phase6_scrape.py  —  Scrape mushroom knowledge sources for RAG
==============================================================
Run this FIRST to populate rag_docs/ with text files.
Each source is saved as a separate .txt file.

Usage:
    python phase6_scrape.py
"""

import os
import time
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR  = os.path.join(BASE_DIR, "rag_docs")
os.makedirs(RAG_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SOURCES = [
    {
        "name": "tnau_mushroom_diseases",
        "url":  "https://agritech.tnau.ac.in/farm_enterprises/Farm%20enterprises_%20Mushroom_Disease.html",
        "type": "html",
    },
    {
        "name": "ahdb_mushroom_diseases",
        "url":  "https://horticulture.ahdb.org.uk/knowledge-library/mushroom-diseases",
        "type": "html",
    },
    {
        "name": "pmc_fungal_biocontrol",
        "url":  "https://pmc.ncbi.nlm.nih.gov/articles/PMC8000694/",
        "type": "html",
    },
    {
        "name": "mushroomworld_species",
        "url":  "https://www.mushroom.world/mushrooms/list",
        "type": "html",
    },
    {
        "name": "fungiatlas_species",
        "url":  "https://fungiatlas.com/all-mushrooms/",
        "type": "html",
    },
    {
        "name": "mushroominfo_encyclopedia",
        "url":  "https://www.mushroominfo.org/encyclopedia/index.html",
        "type": "html",
    },
    {
        "name": "ri_mycological_resources",
        "url":  "https://www.rhodeislandmycologicalsociety.org/blog/resources",
        "type": "html",
    },
    {
        "name": "nios_disease_management",
        "url":  "https://nios.ac.in/media/documents/vocational/mushroom_production_(revised)(618)/Lesson-09.pdf",
        "type": "pdf",
    },
]

# ── Manual expert knowledge ────────────────────────────────────────────────────
# Written specifically for this project — oyster mushroom cultivation in Bangladesh.
# This is the most relevant source for disease classification questions.
MANUAL_DOCS = [
    {
        "name": "oyster_mushroom_diseases_expert",
        "text": """
OYSTER MUSHROOM DISEASE EXPERT KNOWLEDGE
=========================================

ABOUT OYSTER MUSHROOMS
Oyster mushrooms (Pleurotus ostreatus) are widely cultivated edible fungi grown on substrate bags
(paddy straw, sawdust, rice bran). Major cultivation center in Bangladesh is the Mushroom Development
Institute (MDI), Savar, Dhaka, established in 1985. Annual production in Bangladesh exceeds 10,000
metric tons. Oyster mushrooms grow in 22-28°C with 80-90% relative humidity.

DISEASE CLASSIFICATION — THREE CATEGORIES

1. HEALTHY SUBSTRATE BAGS
Visual: Uniform white, cottony mycelium spreading across substrate. No discoloration, no odor.
Action: Continue standard cultivation. Maintain: Temp 22-28°C, Humidity 80-90%, CO2 < 1000 ppm.

2. SINGLE INFECTED SUBSTRATE BAGS
One type of contaminating mold present.

Common single pathogens:
  a) Trichoderma harzianum / T. viride (GREEN MOLD — most common)
     - Appearance: Green powdery patches on substrate
     - Spread: Airborne spores, contaminated spawn, unsterilized substrate
     - Impact: Most destructive contaminant; completely overpowers oyster mycelium
     - Management: Immediate isolation, review sterilization (121°C, 90 min), certified spawn

  b) Aspergillus spp. (BLACK / YELLOW MOLD)
     - Appearance: Black, dark green, or yellow powdery growth
     - Risk: Produces aflatoxins (health hazard)
     - Conditions: High temperature (>30°C), poor humidity control
     - Management: Isolate, discard, improve temperature management

  c) Rhizopus stolonifer (SOFT ROT / BLACK PIN MOLD)
     - Appearance: White cottony growth turning black at tips; substrate becomes slimy
     - Conditions: Excess moisture, poor drainage
     - Management: Reduce substrate moisture, improve drainage, discard affected bags

  d) Neurospora crassa (RED / ORANGE BREAD MOLD)
     - Appearance: Bright orange-pink rapidly spreading growth
     - Risk: Can spread to entire cultivation room within hours
     - Management: Immediate quarantine, remove from facility, bleach treatment of room

3. MIXED INFECTED SUBSTRATE BAGS
Multiple contamination types, or partial contamination with remaining healthy mycelium.
Visual: Multiple color patches, complex visual patterns, combination of white and colored regions.
Action: Remove immediately. Disinfect room with 2% formalin solution. Review entire cultivation chain.

DISEASE PREVENTION PROTOCOL
1. Substrate sterilization: Autoclave at 121°C, 15 psi for 90 minutes minimum
2. Cooling: Cool substrate to < 28°C in a clean room before spawning
3. Spawn quality: Use certified virus-free spawn; inspect before use
4. Positive air pressure: Clean rooms should have positive pressure with HEPA-filtered air
5. Worker hygiene: 70% alcohol spray on gloves/tools; masks and clean clothing mandatory
6. Water: Use clean water; avoid tap water with chloramine in final stages
7. Environmental monitoring: Temperature and humidity sensors; CO2 < 1000 ppm
8. Inspection schedule: Daily visual inspection of all bags; remove contaminated bags within 2 hours
9. Disinfection: 2% formalin spray or 70% isopropyl alcohol between batches
10. Quarantine: Separate room for contaminated bags before disposal

CULTIVATION STAGES
Stage 1 — Substrate Preparation: Mix paddy straw/sawdust + rice bran (10-20%) + lime (1-2%)
Stage 2 — Sterilization: Autoclave 121°C/90 min or pasteurize 80°C/4 hours
Stage 3 — Spawning: Mix spawn into cooled substrate at 5-10% of substrate weight; fill bags
Stage 4 — Spawn Run: 12-20 days at 22-25°C in dark; no light, minimal disturbance
Stage 5 — Pinhead Initiation: Lower temp to 18-22°C, add light 12h/day, increase humidity to 90%
Stage 6 — Fruiting: 5-7 days; harvest when cap edges are still curled (before spore release)
Stage 7 — Subsequent flushes: 3-5 flushes per bag; rest 7-10 days between flushes

NUTRITIONAL PROFILE (Oyster mushroom, dry weight)
Protein: 15-30% | Carbohydrate: 50-65% | Fat: 1-3% | Fiber: 7-12%
Key nutrients: B vitamins (B1, B2, B3, B12), Vitamin D, Potassium, Selenium
Bioactive: Beta-glucans (immune modulation), Lovastatin (cholesterol reduction), Ergothioneine (antioxidant)
Calories: ~33 per 100g fresh weight
""",
    },
    {
        "name": "mushroom_diseases_comprehensive",
        "text": """
COMPREHENSIVE MUSHROOM DISEASE GUIDE
=====================================

FUNGAL DISEASES OF CULTIVATED MUSHROOMS

1. GREEN MOLD (Trichoderma species)
Most economically important disease worldwide.
Pathogens: T. harzianum, T. aggressivum, T. viride, T. koningii
Host: All cultivated mushrooms; oyster, button, shiitake
Symptoms: Green powdery sporulation on substrate and fruiting bodies
Favorable conditions: Temperature 25-30°C, high humidity, high CO2, poor sanitation
Disease cycle: Spores → germination → colonization → sporulation → spread via air/water/tools
Economic loss: Can cause 50-100% crop loss in severe cases
Control:
  - Chemical: Thiabendazole, Prochloraz, Carbendazim (use with caution)
  - Biological: Bacillus subtilis, Trichoderma atroviride (paradoxically used as biocontrol)
  - Cultural: Proper sterilization, clean rooms, HEPA filtration, hygiene protocols

2. WET BUBBLE DISEASE (Mycogone perniciosa)
Pathogen: Mycogone perniciosa (fungus)
Hosts: Button mushroom (Agaricus bisporus) primarily
Symptoms: White fluffy mass at base; malformed, blob-like fruiting bodies; foul smell when mature
Source: Contaminated casing soil, water, workers' clothing
Control: Pasteurize casing soil (60°C, 8 hours), dicloran or thiabendazole spray

3. DRY BUBBLE DISEASE (Lecanicillium fungicola)
Pathogen: Lecanicillium fungicola (formerly Verticillium)
Symptoms: Small, dry, distorted caps; brown marks on stalk; "drumstick" malformation
Conditions: Fluctuating temperatures, excess humidity during pinning
Control: Stable environmental conditions, thiabendazole spray to casing

4. COBWEB DISEASE (Cladobotryum mycophilum)
Pathogen: Cladobotryum mycophilum
Symptoms: Coarse, gray cobweb-like mycelium spreading over caps; sunken brown necrotic spots
Conditions: High humidity (>95%), poor air circulation
Control: Reduce humidity to 85%, increase ventilation, salt or lime application

5. BACTERIAL BLOTCH (Pseudomonas tolaasii)
Pathogen: Pseudomonas tolaasii, P. reactans (bacteria)
Symptoms: Yellow-brown blotches on cap surface; water-soaked lesions; pitting
Conditions: Free water on cap surface, high temperature
Control: Avoid wetting caps, improve air circulation, reduce temperature, copper-based bactericides

6. MUSHROOM VIRUS X (La France Disease)
Pathogen: Multiple mycoviruses (dsRNA viruses)
Symptoms: Thin elongated stalks, small abnormal caps, abnormal gills, crop failure
Spread: Infected spawn, spores, physical contact, insects
Control: No treatment; prevention only — use certified virus-free spawn, strict hygiene

7. BLOTCH DISEASES (various)
Pathogens: Lecanicillium, Verticillium, Dactylium
Symptoms: Brown spots, cap deformities, stalk lesions
Control: Reduce humidity, improve air movement, remove affected fruiting bodies

BACTERIAL DISEASES

Bacterial Pit (Pseudomonas reactans)
Symptoms: Small pits on cap with brown edges; yellow halo
Control: Reduce surface moisture, copper hydroxide spray

Blotch Complex
Multiple Pseudomonas species in combination
Symptoms: Irregular brown to black blotches
Control: Strict hygiene, controlled watering, bactericide application

PHYSIOLOGICAL (ABIOTIC) DISORDERS

1. CO2 Injury: Long stalks, small caps — reduce CO2 with ventilation
2. Temperature Stress: Brown cap, cracked surface — maintain stable temperature
3. Low Humidity: Cracked, split caps — increase misting frequency
4. Chemical Injury: Chlorine from tap water, cleaning agents — use distilled/filtered water
5. Light Deficiency: Etiolated fruiting bodies — ensure 12h light/day

INTEGRATED PEST MANAGEMENT (IPM) FOR MUSHROOMS
1. Prevention-first approach — hygiene and environmental control
2. Monitoring — daily inspection, record keeping
3. Physical control — remove contaminated bags, clean rooms
4. Biological control — beneficial microorganisms
5. Chemical control — last resort, food-safe fungicides only
6. Post-harvest — sterilize rooms between growing cycles
""",
    },
    {
        "name": "mushroom_species_guide",
        "text": """
MUSHROOM SPECIES GUIDE — MAJOR CULTIVATED AND WILD SPECIES
===========================================================

COMMERCIALLY CULTIVATED SPECIES

1. Oyster Mushroom (Pleurotus ostreatus)
   Family: Pleurotaceae | Origin: Temperate forests worldwide
   Substrate: Paddy straw, sawdust, cotton waste, newspaper
   Temperature: 18-28°C (spawn run), 15-22°C (fruiting)
   Yield: 3-5 flushes; 1 kg mushroom per kg substrate
   Uses: Culinary (stir-fry, soup), medicinal (antioxidant, antitumor)
   Varieties: Grey, Golden (P. citrinopileatus), Pink (P. djamor), King (P. eryngii)

2. Button Mushroom (Agaricus bisporus)
   Family: Agaricaceae | Most widely cultivated globally (40% of world production)
   Substrate: Composted horse manure + wheat straw (Phase I and II composting)
   Temperature: 22-25°C (spawn run), 15-18°C (fruiting)
   Main diseases: Wet bubble, dry bubble, cobweb, bacterial blotch

3. Shiitake (Lentinula edodes)
   Family: Omphalotaceae | Origin: East Asia
   Substrate: Oak, cherry hardwood logs or sawdust blocks
   Temperature: 22-26°C (spawn run), 12-18°C (fruiting); requires cold shock
   Medicinal: Lentinan (beta-glucan) with anti-cancer properties, Eritadenine (cholesterol reduction)

4. Enoki (Flammulina velutipes)
   Family: Physalacriaceae
   Substrate: Sawdust with rice bran; grown in cold conditions
   Temperature: 18-22°C (spawn run), 5-10°C (fruiting) — cold fruiting is key characteristic
   Appearance: Long thin stalks, tiny white caps (cultivated); orange-brown stalks (wild)

5. Lion's Mane (Hericium erinaceus)
   Family: Hericiaceae
   Substrate: Hardwood sawdust, supplemented blocks
   Medicinal: Hericenones and erinacines — nerve growth factor (NGF) stimulants
   Uses: Neurological health, cognitive function

6. King Oyster / Trumpet Royale (Pleurotus eryngii)
   Family: Pleurotaceae
   Substrate: Sawdust, cottonseed hulls
   Temperature: 18-24°C; needs CO2 increase for elongated stalk development
   Market value: High; prized for firm texture

MEDICINAL MUSHROOMS

7. Reishi (Ganoderma lucidum)
   Bioactive: Ganoderic acids (triterpenes), beta-glucans, polysaccharides
   Uses: Immune modulation, anti-inflammatory, blood pressure regulation

8. Turkey Tail (Trametes versicolor)
   Bioactive: PSK (Krestin) — approved anti-cancer drug in Japan
   Colorful fan-shaped brackets; grows on dead hardwood

9. Chaga (Inonotus obliquus)
   Grows on birch trees; charcoal-black exterior
   Rich in betulinic acid, antioxidants, melanin

POISONOUS MUSHROOMS (identification warning)

10. Death Cap (Amanita phalloides)
    Most lethal mushroom; causes 90% of fatal mushroom poisonings worldwide
    Contains: Amatoxins (alpha-amanitin) — no antidote, causes liver/kidney failure
    Appearance: Pale greenish cap, white gills, ring and volva at base

11. Destroying Angel (Amanita bisporigera)
    Pure white; contains same amatoxins as Death Cap
    Often confused with edible species by novices

12. Webcaps (Cortinarius species)
    Red-brown caps with cobweb veil; delayed toxicity (nephrotoxic orellanine)
    Symptoms appear 2-3 weeks after ingestion

MUSHROOM ECOLOGY
- Saprotrophic: Decompose dead organic matter (oyster, shiitake, button)
- Mycorrhizal: Symbiotic with tree roots (truffles, chanterelles, porcini)
- Parasitic: Attack living plants or insects (Ophiocordyceps on insects)

SPORE CHARACTERISTICS
- Oyster mushroom: White spores, 7-12 μm, smooth, cylindrical
- Button: Pink spores (young), dark brown-purple (mature), ellipsoid
- Shiitake: White spores, 5-7 μm, smooth, ovoid
""",
    },
]


# ── Scrapers ─────────────────────────────────────────────────────────────────

def scrape_html(url, name):
    """Fetch and clean text from an HTML page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # Try to get main content area
        content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"id": "content"})
            or soup.find("div", {"class": "content"})
            or soup.find("div", {"id": "main-content"})
            or soup.body
        )

        if content is None:
            return None

        text  = content.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text  = "\n".join(lines)

        # Cap at 80,000 chars to avoid huge files
        return text[:80000]

    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return None


def scrape_pdf(url, name):
    """Download PDF and extract text."""
    try:
        import io
        import PyPDF2

        resp = requests.get(url, headers=HEADERS, timeout=40)
        resp.raise_for_status()

        reader = PyPDF2.PdfReader(io.BytesIO(resp.content))
        pages  = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages).strip()

    except ImportError:
        print(f"  [SKIP] {name}: install PyPDF2 with: pip install PyPDF2")
        return None
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PHASE 6A — SCRAPING MUSHROOM KNOWLEDGE BASE")
    print("=" * 65)
    print(f"  Output: {RAG_DIR}\n")

    success, fail = 0, 0

    # 1) Save manual expert documents
    for doc in MANUAL_DOCS:
        path = os.path.join(RAG_DIR, f"{doc['name']}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc["text"].strip())
        print(f"  [OK-manual]  {doc['name']}.txt  ({len(doc['text']):,} chars)")
        success += 1

    print()

    # 2) Scrape online sources
    for src in SOURCES:
        name = src["name"]
        url  = src["url"]
        print(f"  Scraping: {name} ...")

        text = scrape_pdf(url, name) if src["type"] == "pdf" else scrape_html(url, name)

        if text and len(text) > 300:
            path = os.path.join(RAG_DIR, f"{name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"SOURCE: {url}\n{'='*60}\n\n{text}")
            print(f"  [OK]         {name}.txt  ({len(text):,} chars)")
            success += 1
        else:
            print(f"  [SKIP]       {name}: insufficient content or blocked")
            fail += 1

        time.sleep(1.5)  # polite delay

    print()
    print(f"  Done: {success} saved  |  {fail} failed/skipped")
    print(f"  Next step: python phase6_index.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
