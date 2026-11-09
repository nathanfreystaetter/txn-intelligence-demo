from . import report, synth

if __name__ == "__main__":
    print(report.format_report(report.run(synth.generate())))
