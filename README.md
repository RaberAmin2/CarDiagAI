# CarDiagAI

CarDiagAI ("DiaKari") ist ein Streamlit-Prototyp für KI-gestützte Fahrzeugdiagnosen. Die Anwendung verknüpft mehrere Agenten zu einer Analyse-Pipeline, stellt die Ergebnisse in einer Weboberfläche dar und erlaubt eine weiterführende Chat-Interaktion.

## Installation

1. Stelle sicher, dass Python 3.10 oder neuer installiert ist.
2. Führe das Installationsskript aus, um alle Abhängigkeiten einzurichten:

   ```bash
   ./install_dependencies.sh
   ```

   Das Skript verwendet `python -m pip`, um die Pakete aus `requirements.txt` zu installieren und kann in virtuellen Umgebungen ebenso genutzt werden.

## Anwendung starten

Nach der Installation kann die Streamlit-App mit folgendem Befehl gestartet werden:

```bash
streamlit run diagnostic_agent.py
```

Standardmäßig wird der Testmodus aktiviert und ein Beispieltext aus `test_text.txt` geladen. Die Diagnoseergebnisse erscheinen in der Weboberfläche; zusätzlich werden Vorgänge im Log `diagnostic_agent.log` dokumentiert.

## Automatische Versionierung

Das Repository enthält ein einfaches, aber wirkungsvolles Versionierungswerkzeug:

- Die aktuelle Versionsnummer wird in der Datei `VERSION` gespeichert.
- `version_manager.py` stellt Hilfsfunktionen bereit, um die Version zu lesen oder zu erhöhen.
- Über die Kommandozeile kann die Version automatisch angehoben werden:

  ```bash
  python version_manager.py patch
  python version_manager.py minor
  python version_manager.py major
  ```

Jeder Aufruf aktualisiert die `VERSION`-Datei und protokolliert die neue Nummer beim nächsten Start der App. Die Oberfläche zeigt die aktuelle Versionsnummer unter dem Titel an.

### Empfohlener Workflow

1. Implementiere deine Änderungen.
2. Erhöhe die Versionsnummer passend zum Umfang der Änderung.
3. Committe sowohl deinen Code als auch die geänderte `VERSION`-Datei.

Dadurch bleibt nachvollziehbar, welche Funktionalität mit welcher Version veröffentlicht wurde.

## Tests

Die Anwendung nutzt Streamlit und mehrere KI-Agenten. Für schnelle Syntax-Prüfungen kann der folgende Befehl verwendet werden:

```bash
python -m compileall diagnostic_agent.py version_manager.py
```

Weitere Tests sind abhängig von der jeweiligen Infrastruktur und den verwendeten Sprachmodellen.
