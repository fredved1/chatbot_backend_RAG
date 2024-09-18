# LLM Motor: Intelligent Conversatie-agent

## Overzicht

LLM Motor is een geavanceerde conversatie-agent die gebruik maakt van Large Language Models (LLMs) om intelligente en contextbewuste antwoorden te genereren. De agent is specifiek ontworpen voor UWV-gerelateerde vragen en maakt gebruik van verschillende geheugentypes om de context van gesprekken te behouden.

## Hoofdfunctionaliteiten

1. Contextbewuste antwoorden genereren
2. Verschillende geheugentypes voor optimale prestaties
3. Aanpasbare LLM-modellen en parameters

## Installatie

```
## Geheugentypes

### 1. ConversationBufferMemory

- **Beschrijving**: Slaat de volledige gespreksgeschiedenis op.
- **Gebruik**: Standaard geheugentype in `LLMMotor`.
- **Voordelen**: Biedt volledige context voor elk antwoord.
- **Nadelen**: Kan inefficiënt worden bij lange gesprekken.

### 2. ConversationSummaryMemory

- **Beschrijving**: Vat de conversatie samen naarmate deze vordert.
- **Gebruik**: Optioneel, kan worden ingesteld in de constructor.
- **Voordelen**: Efficiënt voor lange gesprekken, behoudt kernpunten.
- **Nadelen**: Kan details missen die later relevant kunnen zijn.

### 3. ConversationKGMemory

- **Beschrijving**: Bouwt een kennisgraaf op basis van de conversatie.
- **Gebruik**: Geavanceerde optie, moet handmatig worden geïmplementeerd.
- **Voordelen**: Uitstekend voor het vastleggen van relaties en entiteiten.
- **Nadelen**: Complexer om te implementeren en te onderhouden.

## Codestructuur

- `llm_motor.py`: Hoofdklasse voor de LLM Motor.
- `templates/uwv_agent.py`: Definieert de UWV-specifieke agent en prompts.
- `memory/`: Map met aangepaste geheugentypes.

## Aanpassen van het Geheugen

Om een ander geheugentype te gebruiken:

```
## Bijdragen

Bijdragen zijn welkom! Zie `CONTRIBUTING.md` voor details.

## Licentie

Dit project is gelicentieerd onder de MIT License - zie het `LICENSE` bestand voor details.
