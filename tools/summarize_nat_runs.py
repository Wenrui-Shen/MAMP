"""Summarize PT -> LP logs without comparing differently scaled PT losses."""
import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = ('motion_patch', 'joint_patch', 'joint_nat', 'joint_nat_cl')


def read_log(path):
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as error:
            raise ValueError('{}:{}: invalid JSON log'.format(path, line_number)) from error
    epochs = [record['epoch'] for record in records]
    if epochs != list(range(len(epochs))):
        raise ValueError('{}: expected unique contiguous epochs from 0'.format(path))
    return records


def summarize_run(name, root):
    pt = read_log(root / name / 'pt' / 'log.txt')
    lp = read_log(root / name / 'lp' / 'log.txt')
    result = dict(experiment=name, source='current_run' if pt or lp else 'not_run',
                  pt_epochs=len(pt), lp_epochs=len(lp), best_lp_acc1='',
                  best_epoch='', last_lp_acc1='')
    if lp:
        best = max(lp, key=lambda item: item['test_acc1'])
        result.update(best_lp_acc1=best['test_acc1'], best_epoch=best['epoch'],
                      last_lp_acc1=lp[-1]['test_acc1'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'experiments/nat/baseline_summary.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [summarize_run(name, args.root) for name in EXPERIMENTS]
    if args.baseline.exists() and rows[0]['source'] == 'not_run':
        baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
        rows[0].update(source='supplied_baseline_logs',
                       pt_epochs=baseline['pretraining']['epochs'],
                       lp_epochs=baseline['linear_probe']['epochs'],
                       best_lp_acc1=baseline['linear_probe']['best_acc1'],
                       best_epoch=baseline['linear_probe']['best_epoch'],
                       last_lp_acc1=baseline['linear_probe']['last_acc1'])
    output = args.output or args.root / 'summary.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print('{experiment}: PT={pt_epochs}, LP={lp_epochs}, best={best_lp_acc1}, '
              'best_epoch={best_epoch}, last={last_lp_acc1}, source={source}'.format(**row))
    print('Saved', output)


if __name__ == '__main__':
    main()
