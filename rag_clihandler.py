class CLIHandler:
    def __init__(self, config_obj: RAGConfig, search_engine_obj: RAGSearchEngine, search_integration_obj: SearchIntegration):
        self.config = config_obj
        self.search_engine = search_engine_obj
        self.search_integration = search_integration_obj
        self.args = None # 파싱된 인자를 저장하기 위함

    # 명령줄 인터페이스
    def parse_arguments():
        """명령줄 인자 파싱"""
        parser = argparse.ArgumentParser(description="RAG 검색 시스템")
        parser.add_argument("-q", "--query", help="검색 쿼리")
        parser.add_argument("-k", "--top-k", type=int, help="반환할 최대 결과 수")
        parser.add_argument("--no-images", action="store_true", help="이미지 결과 제외")
        parser.add_argument("--exact-match", action="store_true", help="정확한 텍스트 매칭만 사용")
        parser.add_argument("--media-type", choices=["image", "text"], help="검색할 미디어 타입 (image 또는 text)")
        parser.add_argument("-i", "--interactive", action="store_true", help="대화형 모드 실행")
        parser.add_argument("--list-metadata", action="store_true", help="인덱스에 포함된 메타데이터 목록 출력")
        parser.add_argument("--no-ai", action="store_true", help="AI 응답 생성 비활성화")
        
        return parser.parse_args()

    # 명령줄 인터페이스 함수
    def cli_search(query: str, top_k: int = None, show_images: bool = True, use_exact_match: bool = False, media_type: str = None, generate_answer: bool = True):
        """명령줄에서 검색을 수행하는 함수
        
        Args:
            query: 검색 쿼리 문자열
            top_k: 반환할 최대 결과 수
            show_images: 이미지 정보 출력 여부
            use_exact_match: 정확한 텍스트 매칭 사용 여부
            media_type: 미디어 타입 필터링 ('image', 'text', None)
        
        Returns:
            검색 결과
        """
        # 검색 통합
        search_integration = SearchIntegration(search_engine)
        
        if use_exact_match:
            # 정확한 텍스트 매칭 검색 수행
            print("\n정확한 텍스트 매칭 검색 수행 중...")
            results = exact_match_search(query, top_k or 5)
            
            # 미디어 타입 필터링 적용
            if media_type:
                filtered_results = []
                for r in results:
                    if media_type == "image" and "file_path" in r and r["file_path"] and \
                    any(r["file_path"].lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]):
                        filtered_results.append(r)
                    elif media_type == "text" and ("file_path" not in r or not r["file_path"] or \
                        not any(r["file_path"].lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"])):
                        filtered_results.append(r)
                results = filtered_results
            
            # 이미지 정보 추출
            images = []
            for r in results:
                if "file_path" in r and r["file_path"]:
                    is_image = any(r["file_path"].lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"])
                    if is_image and (media_type == "image" or media_type is None):
                        images.append({
                            "url": r.get("file_path", ""),
                            "caption": r.get("caption", r.get("text", "이미지 캡션 없음")),
                            "title": r.get("title", ""),
                            "page_num": r.get("page_num", ""),
                            "similarity": r.get("similarity", 0.0)
                        })
            
            response = {
                "results": results,
                "formatted_results": format_search_results(results),
                "has_image": len(images) > 0,
                "images": images,
                "media_type": media_type or "all"
            }
        else:
            # 임베딩 기반 검색 수행
            response = search_integration.process_query(query, media_type=media_type)
        
        # 결과 출력
        print(f"\n검색어: '{query}'\n")
        
        # 미디어 타입 정보 출력
        if media_type:
            print(f"미디어 타입 필터: {media_type}")
        
        # 생성된 응답이 있으면 출력
        if generate_answer and 'generated_answer' in response and response['generated_answer']:
            print("\n===== Ollama 생성 응답 =====")
            print(response['generated_answer'])
            print("===========================\n")
        
        # 검색 결과 출력
        if response["results"]:
            print(response["formatted_results"])
        else:
            print("검색 결과가 없습니다.")
        
        # 이미지 정보 출력 (선택적)
        if show_images and response['has_image']:
            print(f"\n이미지 정보:")
            for i, img in enumerate(response['images']):
                similarity = f"(유사도: {img.get('similarity', 0.0):.4f})" if 'similarity' in img else ""
                print(f"이미지 {i+1}: {img['url']} - {img['title']} (p.{img['page_num']}) {similarity}")
        
        return response

    # 대화형 모드 함수
    def interactive_mode():
        """대화형 모드로 검색 실행"""
        config = RAGConfig()
        search_engine = RAGSearchEngine(config)
        integration = SearchIntegration(search_engine)
        
        print("\n=== RAG 대화형 검색 모드 ===")
        print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
        print("이미지만 검색하려면 'image:'를 쿼리 앞에 추가하세요.")
        print("텍스트만 검색하려면 'text:'를 쿼리 앞에 추가하세요.")
        print("응답 생성을 끄려면 'noai:'를 쿼리 앞에 추가하세요.")
        
        while True:
            try:
                query = input("\n검색어를 입력하세요: ")
                
                # 종료 명령 확인
                if query.lower() in ['exit', 'quit', '종료']:
                    print("검색을 종료합니다.")
                    break
                
                # 미디어 타입 및 응답 생성 옵션 확인
                media_type = None
                generate_answer = True
                
                if query.startswith('image:'):
                    media_type = 'image'
                    query = query[6:].strip()
                elif query.startswith('text:'):
                    media_type = 'text'
                    query = query[5:].strip()
                elif query.startswith('noai:'):
                    generate_answer = False
                    query = query[5:].strip()
                
                # 쿼리가 비어있으면 건너뛰기
                if not query.strip():
                    print("검색어를 입력해주세요.")
                    continue
                
                # 검색 수행
                results = integration.process_query(query, media_type)
                
                # 생성된 응답이 있으면 출력
                if generate_answer and 'generated_answer' in results and results['generated_answer']:
                    print("\n===== Ollama 생성 응답 =====")
                    print(results['generated_answer'])
                    print("===========================\n")
                
                # 결과 출력
                print(f"\n검색 결과 ({results['total_results']}개):")
                for i, result in enumerate(results['results'], 1):
                    print(f"\n[{i}] {result.get('title', '제목 없음')}")
                    
                    # 페이지 정보 출력
                    if 'page_num' in result and result['page_num']:
                        print(f"페이지: {result['page_num']}")
                    
                    # 이미지 경로 출력
                    if 'file_path' in result and result['file_path']:
                        print(f"파일: {result['file_path']}")
                    
                    # 텍스트 내용 출력 (최대 200자)
                    if 'text' in result and result['text']:
                        text = result['text']
                        if len(text) > 200:
                            text = text[:197] + "..."
                        print(f"내용: {text}")
                    
                    # 태그 출력
                    if 'tags' in result and result['tags']:
                        print(f"태그: {', '.join(result['tags'])}")
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {str(e)}")

# 전역 인스턴스 (FastAPI 등을 위해 이미 설정되어 있을 가능성이 높음)
config = RAGConfig()
search_engine = RAGSearchEngine(config) # RAGSearchEngine이 완전히 초기화되었는지 확인
search_integration = SearchIntegration(search_engine) # search_engine 객체를 전달

# FastAPI 앱 정의 (app = FastAPI(...)) 및 엔드포인트들은 대부분 그대로 유지되며,
# 이 전역 인스턴스들을 계속 사용하거나 관리하는 인스턴스를 사용합니다.

if __name__ == "__main__":
    cli_handler = CLIHandler(config, search_engine, search_integration)
    args = cli_handler.parse_arguments()

    if args.list_metadata:
        search_engine.display_metadata_summary() # 새로 만든 메서드 호출
    elif args.interactive:
        cli_handler.start_interactive_mode()
    elif args.query:
        cli_handler.run_cli_search(
            query=args.query,
            top_k=args.top_k,
            show_images=not args.no_images,
            use_exact_match=args.exact_match,
            media_type=args.media_type,
            generate_answer=not args.no_ai
        )
    else:
        # CLI 인자가 없을 때 기본 동작 (예: 도움말 출력 또는 FastAPI 실행 의도 시)
        print("RAG 시스템입니다. 커맨드 라인 옵션은 --help를 사용하세요.")
        # 만약 CLI 인자 없이 실행 시 FastAPI 서버를 시작하고 싶다면 (uvicorn 임포트 필요):
        # print("CLI 인자가 제공되지 않았습니다. FastAPI 서버를 시작합니다...")
        # import uvicorn
        # uvicorn.run(app, host="0.0.0.0", port=8000) # 또는 설정 파일의 포트 사용