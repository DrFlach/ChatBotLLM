# Chatbot LLM + RAG dla programu studiów

## Tytuł projektu

**Chatbot LLM + RAG dla programu studiów**

Projekt przedstawia prototyp systemu typu chatbot, ktory odpowiada na pytania dotyczace programu studiow na podstawie dokumentow uczelni. System wykorzystuje podejscie Retrieval-Augmented Generation, czyli laczy wyszukiwanie informacji w dokumentach z generowaniem odpowiedzi przez model jezykowy.

## Cel projektu

Celem projektu jest przygotowanie dzialajacego MVP aplikacji, ktora umozliwia studentom i kandydatom zadawanie pytan w jezyku naturalnym dotyczacych programu studiow. Aplikacja ma odpowiadac na pytania zwiazane z przedmiotami, sylabusami, terminami egzaminow, prowadzacymi, konsultacjami oraz zasadami zaliczania.

Waznym zalozeniem projektu jest to, aby odpowiedzi byly generowane na podstawie dostarczonych dokumentow, a nie wyłącznie z wiedzy ogolnej modelu jezykowego. Dlatego chatbot zwraca rowniez liste zrodel, ktore zostaly uzyte do przygotowania odpowiedzi.

## Opis problemu

Informacje dotyczace programu studiow sa czesto rozproszone w wielu miejscach: w sylabusach, regulaminach, plikach PDF, tabelach CSV, stronach HTML oraz dokumentach tekstowych. Uzytkownik, ktory chce znalezc konkretna informacje, musi zwykle samodzielnie przegladac kilka dokumentow.

Problemem jest rowniez to, ze pytania uzytkownikow sa zadawane w sposob nieustrukturyzowany, na przyklad:

- "Kiedy jest egzamin z baz danych?"
- "Kto prowadzi algorytmy?"
- "What are the passing rules?"

Tradycyjne wyszukiwanie po slowach kluczowych moze nie znalezc odpowiedzi, jezeli pytanie zostanie sformulowane inaczej niz tekst w dokumencie. Z tego powodu projekt wykorzystuje wyszukiwanie semantyczne, ktore porownuje znaczenie pytania i fragmentow dokumentow.

## Wykorzystane technologie

- **Python** - glowny jezyk implementacji.
- **FastAPI** - framework do budowy backendu i endpointow API.
- **FAISS** - lokalna baza wektorowa do szybkiego wyszukiwania semantycznego.
- **sentence-transformers** - biblioteka do tworzenia embeddingow tekstu.
- **Model embeddingowy `paraphrase-multilingual-MiniLM-L12-v2`** - model obslugujacy wiele jezykow, w tym polski i angielski.
- **OpenAI API** - opcjonalne generowanie odpowiedzi przez model LLM, jezeli skonfigurowano klucz API.
- **pypdf** - odczyt dokumentow PDF.
- **BeautifulSoup** - przetwarzanie dokumentow HTML.
- **HTML, CSS, JavaScript** - prosty interfejs webowy.
- **pytest** - testy jednostkowe.

## Przykladowa baza wiedzy

Projekt zawiera realistyczna przykladowa baze wiedzy dla kierunku Informatyka obejmujaca 7 semestrow studiow. Dane w katalogu `data/raw` opisuja 31 przedmiotow, harmonogram egzaminow i zaliczen, konsultacje 10 prowadzacych oraz regulamin studiow.

Przykladowe dokumenty sa zapisane w kilku formatach:

- `sample_sylabusy.csv` - przedmioty, angielskie nazwy, ECTS, godziny, opisy, tematy sylabusow, prowadzacy i metody zaliczenia,
- `sample_terminy_egzaminow.csv` - daty egzaminow i koncowych zaliczen,
- `sample_konsultacje.html` - konsultacje prowadzacych,
- `sample_program_studiow.txt` - opis programu studiow na semestry 1-7,
- `sample_regulamin.txt` oraz `sample_regulamin_studiow.pdf` - zasady zaliczen, egzaminow, poprawek, ECTS, nieobecnosci, plagiatu, praktyk i pracy inzynierskiej.

Ta baza wiedzy jest sztuczna, ale spojna wewnetrznie i przygotowana tak, aby projekt wygladal jak kompletny program studiow podczas prezentacji uczelnianej.

Po ostatniej ingestii baza FAISS zawiera 97 fragmentow utworzonych z 75 rekordow dokumentow. Projekt ma 71 testow automatycznych i dziala rowniez bez `OPENAI_API_KEY`, korzystajac wtedy z lokalnego fallback generatora odpowiedzi opartego na znalezionym kontekście.

## Architektura systemu

System sklada sie z kilku warstw:

```text
Uzytkownik
   |
   v
Interfejs webowy
   |
   v
FastAPI backend: POST /chat
   |
   v
Retriever
   |
   v
FAISS vector database
   |
   v
Generator odpowiedzi
   |
   v
Odpowiedz + zrodla
```

Najwazniejsze katalogi i pliki:

```text
app/
  main.py                 - konfiguracja aplikacji FastAPI
  api/routes.py           - endpoint /chat
  core/config.py          - ustawienia aplikacji
  rag/ingest.py           - wczytywanie dokumentow TXT, CSV, HTML i PDF
  rag/chunker.py          - dzielenie tekstu na fragmenty
  rag/vector_store.py     - zapis i odczyt indeksu FAISS
  rag/retriever.py        - wyszukiwanie semantyczne
  rag/generator.py        - generowanie odpowiedzi
  web/                    - prosty interfejs uzytkownika
data/raw/                 - przykladowe dokumenty uczelni
data/index/               - lokalny indeks FAISS
scripts/ingest_documents.py
tests/                    - testy jednostkowe
```

## Jak działa RAG w tym projekcie

RAG, czyli Retrieval-Augmented Generation, sklada sie z dwoch glownych etapow.

Pierwszy etap to **retrieval**, czyli wyszukanie fragmentow dokumentow najbardziej pasujacych do pytania uzytkownika. Pytanie jest zamieniane na wektor liczbowy za pomoca modelu embeddingowego. Nastepnie system porownuje ten wektor z wektorami fragmentow dokumentow zapisanymi w bazie FAISS.

Drugi etap to **generation**, czyli przygotowanie odpowiedzi. Jezeli ustawiono `OPENAI_API_KEY`, system przekazuje pytanie oraz znalezione fragmenty dokumentow do modelu LLM. Jezeli klucz API nie jest skonfigurowany, aplikacja uzywa prostego mechanizmu fallback, ktory wybiera najbardziej pasujace zdania ze znalezionych fragmentow.

Odpowiedz powinna wynikac tylko z kontekstu zwroconego przez retriever. Dodatkowo API zwraca zrodla, aby uzytkownik mogl sprawdzic, z jakich dokumentow skorzystano.

## Pipeline przetwarzania dokumentów

Przetwarzanie dokumentow odbywa sie w skrypcie:

```bash
python scripts/ingest_documents.py
```

Pipeline sklada sie z nastepujacych krokow:

1. Wczytanie plikow z katalogu `data/raw`.
2. Obsluga formatow `TXT`, `CSV`, `HTML`, `HTM` oraz `PDF`.
3. Zamiana dokumentow na tekst i metadane, na przykład nazwe pliku, numer strony lub numer wiersza CSV.
4. Podzial tekstu na mniejsze fragmenty z uwzglednieniem akapitow i linii.
5. Utworzenie embeddingow dla kazdego fragmentu.
6. Zapis wektorow do indeksu FAISS.
7. Zapis tekstow i metadanych do pliku `data/index/documents.json`.

Dzieki temu aplikacja nie musi przetwarzac dokumentow przy kazdym pytaniu. W czasie dzialania API korzysta juz z gotowego indeksu.

## Baza wektorowa FAISS

FAISS jest lokalna baza wektorowa uzywana do szybkiego wyszukiwania podobnych wektorow. W projekcie kazdy fragment dokumentu jest reprezentowany przez embedding, czyli wektor liczbowy opisujacy znaczenie tekstu.

Podczas zadawania pytania:

1. pytanie uzytkownika jest zamieniane na embedding,
2. FAISS wyszukuje najbardziej podobne embeddingi dokumentow,
3. system pobiera odpowiadajace im fragmenty tekstu,
4. fragmenty sa przekazywane do generatora odpowiedzi.

W projekcie uzywany jest indeks `IndexFlatIP`, a embeddingi sa normalizowane. Pozwala to stosowac podobienstwo semantyczne oparte na iloczynie skalarnym.

## Endpointy API

### `GET /`

Zwraca prosty interfejs webowy chatbota.

### `GET /health`

Endpoint techniczny do sprawdzenia, czy aplikacja dziala.

Przykladowa odpowiedz:

```json
{
  "status": "ok",
  "index_loaded": true,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "vector_store": "FAISS"
}
```

### `GET /config`

Zwraca bezpieczne informacje konfiguracyjne. Endpoint nie ujawnia klucza OpenAI.

Przykladowa odpowiedz:

```json
{
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "vector_database": "FAISS",
  "openai_configured": false,
  "indexed_chunks": 97
}
```

### `POST /chat`

Glowny endpoint chatbota.

Przykladowe zapytanie:

```json
{
  "question": "Jakie przedmioty sa na Informatyce w semestrze 2?"
}
```

Przykladowa odpowiedz:

```json
{
  "answer": "...",
  "sources": [
    {
      "file_name": "sample_program_studiow.txt",
      "document_type": "txt",
      "chunk_number": 0,
      "subject": null,
      "semester": null,
      "score": 0.81,
      "metadata": {
        "source": "sample_program_studiow.txt",
        "document_type": "txt",
        "chunk_id": 0
      },
      "preview": "..."
    }
  ]
}
```

## Przykładowe pytania po polsku i angielsku

Przyklady pytan po polsku:

- Jakie przedmioty sa na Informatyce w semestrze 1?
- Jakie przedmioty sa na Informatyce w semestrze 2?
- Jakie przedmioty sa na Informatyce w semestrze 3?
- Jakie przedmioty sa na Informatyce w semestrze 4?
- Jakie przedmioty sa na Informatyce w semestrze 5?
- Jakie przedmioty sa na Informatyce w semestrze 6?
- Jakie przedmioty sa na Informatyce w semestrze 7?
- Kiedy jest egzamin z Algorytmow i struktur danych?
- Jakie sa konsultacje dr Anny Kowalskiej?
- Jak wyglada zaliczenie przedmiotu Bazy danych?
- Kto prowadzi Uczenie maszynowe?
- Co obejmuje DevOps i CI/CD?
- Jak wyglada zaliczenie Chmury obliczeniowej?
- Kiedy jest obrona Pracy inzynierskiej?
- Jakie sa zasady zaliczenia semestru?

Przyklady pytan po angielsku:

- What courses are included in semester 2 of Computer Science?
- What subjects are included in semester 5?
- What subjects are included in semester 6?
- What subjects are included in semester 7?
- Who teaches Databases?
- Who teaches Machine Learning?
- What is covered in Cloud Computing?
- How is DevOps and CI/CD assessed?
- When is the Engineering Thesis defense?
- What are the passing rules for a semester?
- When is the Algorithms and Data Structures exam?
- What are the consultation hours for Anna Kowalska?

## Zrzuty ekranu

W tej sekcji można dodać zrzuty ekranu z działania aplikacji.

Przykładowe miejsca na obrazy:

```text
docs/screenshots/main_view.png
docs/screenshots/example_answer_pl.png
docs/screenshots/example_answer_en.png
docs/screenshots/api_docs.png
```

Proponowane zrzuty ekranu:

- widok głównego interfejsu chatbota,
- przykładowa odpowiedź w języku polskim,
- przykładowa odpowiedź w języku angielskim,
- dokumentacja Swagger UI pod adresem `/docs`.

## Ograniczenia

- Projekt jest prototypem MVP i korzysta z realistycznego, ale przykladowego zestawu dokumentow dla 7 semestrow Informatyki.
- Jakosc odpowiedzi zalezy od jakosci oraz kompletności dokumentow w katalogu `data/raw`.
- Fallback bez OpenAI API nie generuje pelnych odpowiedzi tak dobrze jak model LLM; wybiera najtrafniejsze fragmenty tekstu.
- Pierwsze uruchomienie ingestii moze wymagac dostepu do internetu, aby pobrac model embeddingowy.
- System nie posiada logowania uzytkownikow ani panelu administracyjnego.
- Aktualizacja dokumentow wymaga ponownego uruchomienia skryptu ingestii.

## Możliwe przyszłe usprawnienia

- Dodanie panelu administracyjnego do przesylania dokumentow.
- Automatyczne odswiezanie indeksu FAISS po dodaniu nowych plikow.
- Lepsza obsluga plikow PDF o skomplikowanym ukladzie.
- Dodanie historii rozmow uzytkownika.
- Dodanie filtrowania wynikow po kierunku studiow, semestrze lub typie dokumentu.
- Rozszerzenie testow integracyjnych dla endpointu `/chat`.
- Dodanie konteneryzacji Docker.
- Dodanie oceny jakosci odpowiedzi i mechanizmu feedbacku od uzytkownika.

## Jak uruchomić projekt

Wymagany jest Python 3.11 lub nowszy.

### 1. Utworzenie środowiska wirtualnego

```bash
python -m venv .venv
source .venv/bin/activate
```

W systemie Windows aktywacja środowiska może wyglądać tak:

```powershell
.venv\Scripts\activate
```

### 2. Instalacja zależności

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Opcjonalna konfiguracja OpenAI

Projekt dziala bez klucza OpenAI, ale wtedy korzysta z fallbacku extractive.

```bash
cp .env.example .env
```

Nastepnie w pliku `.env` mozna ustawic:

```text
OPENAI_API_KEY=your_api_key_here
```

### 4. Zbudowanie indeksu FAISS

```bash
python scripts/ingest_documents.py
```

Po wykonaniu tego polecenia w katalogu `data/index` powinny pojawic sie pliki indeksu.

### 5. Uruchomienie aplikacji

```bash
uvicorn app.main:app --reload
```

Aplikacja bedzie dostepna pod adresem:

```text
http://127.0.0.1:8000
```

Dokumentacja API bedzie dostepna pod adresem:

```text
http://127.0.0.1:8000/docs
```

### 6. Uruchomienie testów

```bash
pytest
```

Testy jednostkowe sprawdzaja m.in. dzielenie tekstu, dzialanie bazy wektorowej oraz fallback generatora odpowiedzi.

## Uruchomienie przez Docker

Projekt mozna uruchomic bez lokalnego instalowania zaleznosci Pythona, korzystajac z Dockera. Obraz buduje indeks FAISS podczas `docker build`, a kontener startuje aplikacje przez `start.sh`.

### Docker

Zbuduj obraz:

```bash
docker build -t chatbot-rag .
```

Uruchom kontener:

```bash
docker run --rm -p 8000:8000 chatbot-rag
```

Aplikacja bedzie dostepna pod adresem:

```text
http://127.0.0.1:8000
```

Dokumentacja API:

```text
http://127.0.0.1:8000/docs
```

### Docker Compose

Alternatywnie mozna uzyc Docker Compose:

```bash
docker compose up --build
```

Jezeli chcesz uzyc OpenAI API, ustaw zmienna srodowiskowa przed uruchomieniem:

```bash
OPENAI_API_KEY=your_api_key_here docker compose up --build
```

Klucz OpenAI API nie jest wymagany. Bez niego projekt dziala w trybie fallback, korzystajac z lokalnego generatora odpowiedzi opartego na znalezionych dokumentach.

Po zmianie plikow w `data/raw` nalezy przebudowac obraz, aby odtworzyc indeks FAISS:

```bash
docker compose build --no-cache
docker compose up
```

Klucz OpenAI API nie jest wymagany do wdrozenia, poniewaz projekt dziala rowniez w trybie fallback bez OpenAI API. Pliki zrodlowe bazy wiedzy w `data/raw` powinny byc commitowane do repozytorium. Katalog `data/index` jest ignorowany w `.gitignore` poza plikiem `.gitkeep`, dlatego na Render indeks jest odtwarzany przez Build Command.

## Rozszerzanie bazy wiedzy

Baze wiedzy mozna rozszerzac bez zmian w kodzie aplikacji. Wystarczy dodac nowe pliki do katalogu `data/raw` i ponownie zbudowac indeks FAISS.

Aby dodac nowe dokumenty do bazy wiedzy:

1. Umiesc pliki w katalogu `data/raw`.
2. Upewnij sie, ze format pliku to `TXT`, `CSV`, `HTML`, `HTM` albo `PDF`.
3. Uruchom ponownie:

```bash
python scripts/ingest_documents.py
```

Po przebudowaniu indeksu chatbot bedzie mogl korzystac z nowych dokumentow.
