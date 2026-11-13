"""Start the packaged local viewer: python -m betalens.experiments serve --db PATH."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Betalens 离线实验档案")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="以只读模式启动本地看板")
    serve.add_argument("--db", required=True, help="现有 LMDB 实验库目录")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    try:
        import uvicorn
        from .api import create_app
    except ImportError as exc:
        parser.error(f"Missing optional dependency: {exc}. Install betalens[experiment-viewer].")
    uvicorn.run(create_app(args.db), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
