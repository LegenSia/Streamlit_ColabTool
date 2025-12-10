import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from contextlib import closing
import psycopg2
from streamlit_calendar import calendar as st_calendar
from st_circular_progress import CircularProgress


# =========================================================
# DB 연결 (Supabase PostgreSQL)
# =========================================================
def get_conn():
    # secrets에 postgres 설정이 없을 때 방어
    if "postgres" not in st.secrets:
        st.error(
            "Postgres 설정이 없습니다. Streamlit Secrets에 [postgres] 정보를 추가하세요."
        )
        st.stop()

    cfg = st.secrets["postgres"]
    return psycopg2.connect(
        host=cfg["host"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        port=cfg.get("port", 5432),
        sslmode="require",  # Supabase는 SSL 필수
    )


# =========================================================
# 초기 스키마 / 샘플 데이터
# =========================================================
def init_db():
    with closing(get_conn()) as conn:
        cur = conn.cursor()

        # projects
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS projects(
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL
        );
        """
        )
        # 스키마 마이그레이션 (옛 테이블에 description 없을 수 있음)
        cur.execute(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS description TEXT;"
        )
        cur.execute(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS created_at TEXT;"
        )

        # parts
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS parts(
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            color TEXT,
            created_at TEXT NOT NULL
        );
        """
        )

        # users
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS users(
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            part_id INTEGER,
            role TEXT,
            created_at TEXT NOT NULL
        );
        """
        )

        # tasks
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS tasks(
            id SERIAL PRIMARY KEY,
            project_id INTEGER NOT NULL,
            part_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            assignee TEXT,
            priority TEXT,
            status TEXT,
            start_date TEXT,
            due_date TEXT,
            progress INTEGER,
            tags TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
        )

        # user_parts
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS user_parts(
            user_id INTEGER NOT NULL,
            part_id INTEGER NOT NULL,
            PRIMARY KEY(user_id, part_id)
        );
        """
        )

        # user_projects
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS user_projects(
            user_id INTEGER NOT NULL,
            project_id INTEGER NOT NULL,
            PRIMARY KEY(user_id, project_id)
        );
        """
        )

        conn.commit()


def seed_if_empty():
    now = datetime.utcnow().isoformat()

    # '데모 프로젝트' 제거
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM tasks WHERE project_id IN (SELECT id FROM projects WHERE name=%s)",
            ("데모 프로젝트",),
        )
        cur.execute("DELETE FROM projects WHERE name=%s", ("데모 프로젝트",))
        conn.commit()

    # 프로젝트 기본 세팅
    projects_df = list_projects()
    if "빈 샘플 프로젝트" not in projects_df["name"].tolist():
        insert_project("빈 샘플 프로젝트", "빈 프로젝트 (테스트용)")

    # 파트 기본 세팅
    default_colors = {
        "기획": "#F97373",
        "개발": "#6CB2EB",
        "아트": "#FBC15E",
    }
    parts_df = list_parts()
    existing_names = parts_df["name"].tolist()

    for name, color in default_colors.items():
        if name not in existing_names:
            insert_part(name, color)

    parts_df = list_parts()
    for _, row in parts_df.iterrows():
        if not isinstance(row.get("color"), str) or not row["color"]:
            color = default_colors.get(row["name"], "#3788d8")
            update_part(int(row["id"]), color=color)

    # 유저 기본 세팅
    users_df = list_users()
    if users_df.empty:
        parts_df = list_parts()
        parts_map = {
            row["name"]: int(row["id"]) for _, row in parts_df.iterrows()
        }
        sample_users = [
            ("기획자 A", "planner@example.com", parts_map.get("기획"), "planner"),
            ("개발자 B", "dev@example.com", parts_map.get("개발"), "developer"),
            ("아티스트 C", "artist@example.com", parts_map.get("아트"), "artist"),
        ]
        for name, email, pid, role in sample_users:
            if pid:
                insert_user(name, email, [pid], role)
            else:
                insert_user(name, email, [], role)

    # 기존 유저들에게 프로젝트 권한 기본 부여
    users_df = list_users()
    projects_df = list_projects()
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        for _, u in users_df.iterrows():
            for _, p in projects_df.iterrows():
                cur.execute(
                    """
                    INSERT INTO user_projects(user_id, project_id)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id, project_id) DO NOTHING
                    """,
                    (int(u["id"]), int(p["id"])),
                )
        conn.commit()

    # 샘플 작업
    tasks_df = list_tasks()
    if tasks_df.empty and not projects_df.empty:
        project_id = int(projects_df.iloc[0]["id"])
        parts_df = list_parts()
        parts_map = {
            row["name"]: int(row["id"]) for _, row in parts_df.iterrows()
        }
        description = "요구사항 수집|40|0\n와이어프레임 정리|60|0"
        insert_task(
            project_id=project_id,
            part_id=parts_map["기획"],
            title="기획 문서 정리",
            description=description,
            assignee="기획자 A",
            priority="High",
            status="Todo",
            start_date=(date.today() - timedelta(days=1)).isoformat(),
            due_date=(date.today() + timedelta(days=2)).isoformat(),
            progress=0,
            tags="기획,문서",
        )


# --- DB 초기화/시드 1회만 실행하도록 래핑 ---
@st.cache_resource(show_spinner=False)
def ensure_db_initialized():
    init_db()
    seed_if_empty()
    return True


# =========================================================
# Data Access (Postgres + pandas)  - 캐시 적용
# =========================================================
@st.cache_data(show_spinner=False)
def list_projects():
    with closing(get_conn()) as conn:
        return pd.read_sql_query(
            "SELECT * FROM projects ORDER BY created_at DESC, id DESC",
            conn,
        )


@st.cache_data(show_spinner=False)
def list_parts():
    with closing(get_conn()) as conn:
        return pd.read_sql_query(
            "SELECT * FROM parts ORDER BY id",
            conn,
        )


@st.cache_data(show_spinner=False)
def list_users():
    with closing(get_conn()) as conn:
        query = """
        SELECT u.*,
               COALESCE(string_agg(p.name, ', ' ORDER BY p.id), '') AS part_names
        FROM users u
        LEFT JOIN user_parts up ON up.user_id = u.id
        LEFT JOIN parts p ON p.id = up.part_id
        GROUP BY u.id
        ORDER BY u.id
        """
        return pd.read_sql_query(query, conn)


@st.cache_data(show_spinner=False)
def list_tasks(project_id=None, part_id=None):
    with closing(get_conn()) as conn:
        base = """
        SELECT t.*, p.name AS part_name, p.color AS part_color
        FROM tasks t
        JOIN parts p ON p.id = t.part_id
        """
        conds = []
        params = []
        if project_id is not None:
            conds.append("t.project_id = %s")
            params.append(project_id)
        if part_id is not None:
            conds.append("t.part_id = %s")
            params.append(part_id)
        if conds:
            base += " WHERE " + " AND ".join(conds)
        base += " ORDER BY t.due_date IS NULL, t.due_date ASC, t.id DESC"
        return pd.read_sql_query(base, conn, params=params)


def insert_project(name, description):
    now = datetime.utcnow().isoformat()
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO projects(name, description, created_at) VALUES(%s,%s,%s)",
            (name, description, now),
        )
        conn.commit()
    # --- 캐시 무효화 ---
    list_projects.clear()


def update_project(project_id, **kwargs):
    sets = []
    params = []
    for k, v in kwargs.items():
        sets.append(f"{k}=%s")
        params.append(v)
    if not sets:
        return
    params.append(project_id)
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE projects SET {', '.join(sets)} WHERE id=%s",
            params,
        )
        conn.commit()
    # --- 캐시 무효화 ---
    list_projects.clear()


def delete_project(project_id):
    """프로젝트 삭제 (관련 tasks, user_projects 정리)"""
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM tasks WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM user_projects WHERE project_id=%s", (project_id,))
        cur.execute("DELETE FROM projects WHERE id=%s", (project_id,))
        conn.commit()
    list_projects.clear()
    list_tasks.clear()
    get_projects_for_user.clear()


def insert_part(name, color="#3788d8"):
    now = datetime.utcnow().isoformat()
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO parts(name, color, created_at) VALUES(%s,%s,%s)",
            (name, color, now),
        )
        conn.commit()
    # --- 캐시 무효화 ---
    list_parts.clear()

def delete_part(part_id):
    """파트 삭제 (연관 작업/유저-파트 매핑도 정리)"""
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        # 이 파트에 속한 작업 삭제
        cur.execute("DELETE FROM tasks WHERE part_id=%s", (part_id,))
        # 유저-파트 매핑 삭제
        cur.execute("DELETE FROM user_parts WHERE part_id=%s", (part_id,))
        # 실제 파트 삭제
        cur.execute("DELETE FROM parts WHERE id=%s", (part_id,))
        conn.commit()

    # 캐시 무효화
    list_parts.clear()
    list_tasks.clear()
    get_parts_for_user.clear()
    get_users_for_part.clear()

def update_part(part_id, **kwargs):
    sets = []
    params = []
    for k, v in kwargs.items():
        sets.append(f"{k}=%s")
        params.append(v)
    if not sets:
        return
    params.append(part_id)
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE parts SET {', '.join(sets)} WHERE id=%s",
            params,
        )
        conn.commit()
    # --- 캐시 무효화 ---
    list_parts.clear()



def insert_user(name, email, part_ids, role):
    now = datetime.utcnow().isoformat()
    main_part_id = part_ids[0] if part_ids else None
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users(name,email,part_id,role,created_at) VALUES(%s,%s,%s,%s,%s) RETURNING id",
            (name, email, main_part_id, role, now),
        )
        res = cur.fetchone()
        user_id = res[0] if res else None

        if user_id is None:
            # 다시 id 가져오기
            cur.execute(
                "SELECT id FROM users WHERE name=%s AND email=%s ORDER BY id DESC LIMIT 1",
                (name, email),
            )
            user_id = cur.fetchone()[0]

        for pid in part_ids or []:
            cur.execute(
                """
                INSERT INTO user_parts(user_id, part_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, part_id) DO NOTHING
                """,
                (user_id, pid),
            )

        cur.execute("SELECT id FROM projects")
        prows = cur.fetchall()
        for p in prows:
            pid = p[0]
            cur.execute(
                """
                INSERT INTO user_projects(user_id, project_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, project_id) DO NOTHING
                """,
                (user_id, pid),
            )
        conn.commit()

    # --- 캐시 무효화 ---
    list_users.clear()
    get_parts_for_user.clear()
    get_users_for_part.clear()
    get_projects_for_user.clear()

    return user_id


def update_user(user_id, **kwargs):
    sets = []
    params = []
    for k, v in kwargs.items():
        sets.append(f"{k}=%s")
        params.append(v)
    if not sets:
        return
    params.append(user_id)
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE users SET {', '.join(sets)} WHERE id=%s",
            params,
        )
        conn.commit()

    # --- 캐시 무효화 ---
    list_users.clear()
    get_parts_for_user.clear()
    get_users_for_part.clear()
    get_projects_for_user.clear()


def delete_user(user_id):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_parts WHERE user_id=%s", (user_id,))
        cur.execute("DELETE FROM user_projects WHERE user_id=%s", (user_id,))
        cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
        conn.commit()

    # --- 캐시 무효화 ---
    list_users.clear()
    get_parts_for_user.clear()
    get_users_for_part.clear()
    get_projects_for_user.clear()


def set_user_parts(user_id, part_ids):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_parts WHERE user_id=%s", (user_id,))
        for pid in part_ids or []:
            cur.execute(
                """
                INSERT INTO user_parts(user_id, part_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, part_id) DO NOTHING
                """,
                (user_id, pid),
            )
        conn.commit()

    # --- 캐시 무효화 ---
    list_users.clear()
    get_parts_for_user.clear()
    get_users_for_part.clear()


@st.cache_data(show_spinner=False)
def get_parts_for_user(user_id):
    with closing(get_conn()) as conn:
        return pd.read_sql_query(
            """
        SELECT p.*
        FROM user_parts up
        JOIN parts p ON p.id = up.part_id
        WHERE up.user_id = %s
        ORDER BY p.id
        """,
            conn,
            params=[user_id],
        )


@st.cache_data(show_spinner=False)
def get_users_for_part(part_id):
    with closing(get_conn()) as conn:
        return pd.read_sql_query(
            """
        SELECT u.*
        FROM users u
        JOIN user_parts up ON up.user_id = u.id
        WHERE up.part_id = %s
        ORDER BY u.id
        """,
            conn,
            params=[part_id],
        )


@st.cache_data(show_spinner=False)
def get_projects_for_user(user_id):
    with closing(get_conn()) as conn:
        return pd.read_sql_query(
            """
        SELECT pr.*
        FROM user_projects up
        JOIN projects pr ON pr.id = up.project_id
        WHERE up.user_id = %s
        ORDER BY pr.created_at DESC, pr.id DESC
        """,
            conn,
            params=[user_id],
        )


def set_user_projects(user_id, project_ids):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_projects WHERE user_id=%s", (user_id,))
        for pid in project_ids or []:
            cur.execute(
                """
                INSERT INTO user_projects(user_id, project_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, project_id) DO NOTHING
                """,
                (user_id, pid),
            )
        conn.commit()

    # --- 캐시 무효화 ---
    get_projects_for_user.clear()


def insert_task(
    project_id,
    part_id,
    title,
    description,
    assignee,
    priority,
    status,
    start_date,
    due_date,
    progress,
    tags,
):
    now = datetime.utcnow().isoformat()
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tasks(
                project_id, part_id, title, description, assignee,
                priority, status, start_date, due_date, progress, tags,
                created_at, updated_at
            ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                project_id,
                part_id,
                title,
                description,
                assignee,
                priority,
                status,
                start_date,
                due_date,
                progress,
                tags,
                now,
                now,
            ),
        )
        conn.commit()

    # --- 캐시 무효화 ---
    list_tasks.clear()


def update_task(task_id, **kwargs):
    now = datetime.utcnow().isoformat()
    sets = []
    params = []
    for k, v in kwargs.items():
        sets.append(f"{k}=%s")
        params.append(v)
    sets.append("updated_at=%s")
    params.append(now)
    params.append(task_id)
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id=%s",
            params,
        )
        conn.commit()

    # --- 캐시 무효화 ---
    list_tasks.clear()


def delete_task(task_id):
    with closing(get_conn()) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM tasks WHERE id=%s", (task_id,))
        conn.commit()

    # --- 캐시 무효화 ---
    list_tasks.clear()


# =========================================================
# Helper: 색상 변형 / 캘린더
# =========================================================
def adjust_color(hex_color: str, index: int) -> str:
    if not isinstance(hex_color, str) or not hex_color:
        hex_color = "#3788d8"
    c = hex_color.lstrip("#")
    if len(c) != 6:
        c = "3788d8"
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    offsets = [-0.4, -0.2, 0.0, 0.2, 0.4]
    factor = 1.0 + offsets[index % len(offsets)]
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02X}{g:02X}{b:02X}"


def build_calendar_events(tasks_df, show_part_in_title=True):
    events = []
    if tasks_df is None or tasks_df.empty:
        return events

    color_idx_by_part = {}

    for _, r in tasks_df.iterrows():
        s = r.get("start_date")
        e = r.get("due_date")

        # 날짜 기본값 보정
        if not s and e:
            s = e
        if not e and s:
            e = s
        if not s and not e:
            s = e = date.today().isoformat()

        # 문자열로 통일
        if not isinstance(s, str):
            try:
                s = s.date().isoformat()
            except Exception:
                s = date.today().isoformat()
        if not isinstance(e, str):
            try:
                e = e.date().isoformat()
            except Exception:
                e = s

        # 🔹 allDay 이벤트는 end가 "전날"까지라서, 캘린더에만 +1일 적용
        try:
            e_date = date.fromisoformat(e)
            e_plus = (e_date + timedelta(days=1)).isoformat()
        except Exception:
            e_plus = e

        title = r["title"]
        if show_part_in_title and isinstance(r.get("part_name"), str):
            title = f"[{r['part_name']}] {title}"

        base_color = (
            r.get("part_color")
            if isinstance(r.get("part_color"), str) and r["part_color"]
            else "#3788d8"
        )

        part_id = r.get("part_id")
        idx = color_idx_by_part.get(part_id, 0)
        color_idx_by_part[part_id] = idx + 1
        color = adjust_color(base_color, idx)

        event = {
            "id": str(r["id"]),
            "title": title,
            "start": s,
            "end": e_plus,          # 캘린더에는 +1일 적용된 값
            "allDay": True,
            "backgroundColor": color,
            "borderColor": color,
            "extendedProps": {
                "assignee": r.get("assignee"),
                "priority": r.get("priority"),
                "status": r.get("status"),
            },
        }
        events.append(event)
    return events

def calendar_options_base():
    return {
        "initialView": "dayGridMonth",
        "headerToolbar": {
            "left": "title",
            "center": "",
            "right": "dayGridMonth,dayGridWeek,dayGridDay prev,next",
        },
        "locale": "ko",
        "selectable": True,
        "editable": False,
        "height": 550,
        "contentHeight": 480,
        "aspectRatio": 1.35,
    }


# =========================================================
# Subtask 파싱 / 진행률 계산
# =========================================================
def parse_subtasks(description: str):
    if not description:
        return []
    lines = description.splitlines()
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            label = parts[0]
            weight = 0
            done = False
            if len(parts) > 1:
                try:
                    weight = int(parts[1])
                except Exception:
                    weight = 0
            if len(parts) > 2:
                try:
                    done = bool(int(parts[2]))
                except Exception:
                    done = False
            result.append((label, max(0, min(100, weight)), done))
            continue

        done = False
        label = line
        if line.startswith("[x] "):
            done = True
            label = line[4:]
        elif line.startswith("[ ] "):
            done = False
            label = line[4:]
        weight = 100
        result.append((label.strip(), weight, done))
    return result


def serialize_subtasks(subtasks):
    lines = []
    for label, weight, done in subtasks:
        lines.append(f"{label}|{int(weight)}|{1 if done else 0}")
    return "\n".join(lines)


def calc_progress_from_subtasks(subtasks):
    if not subtasks:
        return 0
    s = sum(int(w) for _, w, done in subtasks if done)
    return min(100, max(0, s))


def priority_label_and_color(priority: str):
    if priority == "High":
        return "높음", "#FF4B4B"
    if priority == "Low":
        return "낮음", "#4CAF50"
    return "중간", "#FFDD57"


def completion_ratio(tasks_df: pd.DataFrame) -> int:
    if tasks_df is None or tasks_df.empty:
        return 0
    total = len(tasks_df)
    done_equiv = 0.0
    for _, r in tasks_df.iterrows():
        status = r.get("status") or ""
        prog = r.get("progress") or 0
        if status == "Done":
            done_equiv += 1.0
        elif status == "In Progress":
            try:
                done_equiv += float(prog) / 100.0
            except Exception:
                pass
    return int(round(100 * done_equiv / total))


# =========================================================
# 날짜 클릭 값 정규화 (UTC→KST 보정 포함)
# =========================================================
def normalize_clicked_date(dc: dict) -> str | None:
    """
    streamlit-calendar의 dateClick 반환값을
    'YYYY-MM-DD' 문자열로 안전하게 변환.
    KST(+9) 기준으로 하루 밀리는 현상 방지.
    """
    raw = dc.get("dateStr") or dc.get("date") or ""
    if not raw:
        return None
    raw = str(raw)

    try:
        # '2025-12-20' 형태 → 그대로 사용
        if "T" not in raw:
            return raw[:10]

        # '2025-12-19T15:00:00Z' 같은 형태
        s = raw.rstrip("Z")
        dt = datetime.fromisoformat(s)

        # raw가 UTC 기준이라고 가정하고 +9시간 (KST) 보정
        if raw.endswith("Z"):
            dt = dt + timedelta(hours=9)

        return dt.date().isoformat()
    except Exception:
        # 실패하면 앞 10자리만 자르기
        return raw[:10]


# =========================================================
# Fragment: 대시보드
# =========================================================
@st.fragment()
def render_dashboard(selected_project_id, parts_df, part_names, CURRENT_USER, role):
    st.subheader("📊 대시보드 (전체 파트 일정)")

    if not selected_project_id:
        st.info("좌측에서 프로젝트를 선택하세요.")
        return

    col1, col2 = st.columns([3, 1])
    with col2:
        part_filter_name = st.selectbox("파트 필터", ["전체"] + part_names)
    with col1:
        pass

    all_tasks = list_tasks(project_id=selected_project_id)

    if part_filter_name != "전체":
        part_row = parts_df[parts_df["name"] == part_filter_name]
        if not part_row.empty:
            part_id_filter = int(part_row["id"].iloc[0])
            filtered = list_tasks(
                project_id=selected_project_id, part_id=part_id_filter
            )
        else:
            filtered = all_tasks.iloc[0:0]
    else:
        filtered = all_tasks

    # --- 날짜 상태 기본값: 오늘 ---
    key_sel = "dashboard_selected_date"
    if key_sel not in st.session_state:
        st.session_state[key_sel] = date.today().isoformat()
    current_sel_str = st.session_state[key_sel]

    events = build_calendar_events(filtered, show_part_in_title=True)
    options = calendar_options_base()
    # 탭 이동 후에도 캘린더가 마지막 선택 날짜를 기준으로 보이도록
    options["initialDate"] = current_sel_str

    cal_val = st_calendar(
        events=events,
        options=options,
        key="dashboard_calendar",
    )

    # --- 날짜 클릭 콜백 처리 ---
    if isinstance(cal_val, dict) and cal_val.get("callback") == "dateClick":
        dc = cal_val.get("dateClick", {})
        clicked = normalize_clicked_date(dc)
        if clicked:
            st.session_state[key_sel] = clicked
            current_sel_str = clicked

    selected_day_str = current_sel_str
    selected_day = date.fromisoformat(selected_day_str)

    def is_on_day(row):
        """선택한 날짜가 작업 기간(start_date~due_date) 안에 들어가면 True"""

        def _parse(v):
            # 문자열이면 YYYY-MM-DD 로 파싱
            if isinstance(v, str) and v:
                try:
                    return date.fromisoformat(v[:10])
                except Exception:
                    return None
            # pandas Timestamp 등일 수 있는 경우
            try:
                if pd.notna(v):
                    return v.date()
            except Exception:
                return None
            return None

        s = _parse(row.get("start_date"))
        e = _parse(row.get("due_date"))

        # 둘 다 없으면 이 날짜엔 안 보이게
        if s is None and e is None:
            return False
        # 하나만 있으면 그 날 하루짜리로 취급
        if s is None:
            s = e
        if e is None:
            e = s

        return s <= selected_day <= e

    day_tasks = (
        filtered[filtered.apply(is_on_day, axis=1)]
        if not filtered.empty
        else filtered
    )

    if not day_tasks.empty:
        st.markdown(f"#### 선택한 날짜 일정 ({selected_day.isoformat()})")
        show_cols = [
            "id",
            "title",
            "part_name",
            "assignee",
            "status",
            "priority",
            "start_date",
            "due_date",
            "progress",
            "tags",
        ]
        exist_cols = [c for c in show_cols if c in day_tasks.columns]
        st.dataframe(
            day_tasks[exist_cols], use_container_width=True, hide_index=True
        )

    br_col, graph_col = st.columns([2, 2])

    with br_col:
        # user 계정(=기획)일 때만 브리핑 출력
        if role == "user":
            st.markdown("#### 🧍 나의 할 일 브리핑")
            if filtered.empty:
                st.caption("현재 프로젝트에 등록된 작업이 없습니다.")
            else:
                my_tasks = filtered[filtered["assignee"] == CURRENT_USER]
                if my_tasks.empty:
                    st.caption(
                        f"현재 프로젝트/필터에서 {CURRENT_USER}에게 배정된 작업이 없습니다."
                    )
                else:
                    total = len(my_tasks)
                    by_status = my_tasks["status"].value_counts().to_dict()

                    def parse_due(x):
                        try:
                            if isinstance(x, str) and x:
                                return date.fromisoformat(x)
                            if pd.notna(x):
                                return x.date()
                            return None
                        except Exception:
                            return None

                    my_tasks = my_tasks.copy()
                    my_tasks["due_dt"] = my_tasks["due_date"].apply(parse_due)
                    upcoming = my_tasks.dropna(subset=["due_dt"]).sort_values("due_dt")
                    if not upcoming.empty:
                        next_due = upcoming.iloc[0]
                        next_due_date = next_due["due_dt"].isoformat()
                        next_due_title = next_due["title"]
                    else:
                        next_due_date = "-"
                        next_due_title = "-"

                    st.markdown(
                        f"- 총 작업 수: **{total}건**  "
                        f"(Todo: {by_status.get('Todo', 0)}, In Progress: {by_status.get('In Progress', 0)}, Done: {by_status.get('Done', 0)})"
                    )
                    st.markdown(
                        f"- 가장 가까운 마감: **{next_due_date} · {next_due_title}**"
                    )
        # admin일 때는 이 칸 비워두기

    with graph_col:
        st.markdown("#### 전체 / 파트 진행률")
        if all_tasks is None or all_tasks.empty:
            st.caption("진행률 데이터가 없습니다.")
        else:
            overall = completion_ratio(all_tasks)
            items = []

            items.append(
                {
                    "label": "전체",
                    "value": overall,
                    "color": "#4A5568",
                }
            )

            for _, prow in parts_df.iterrows():
                pid = prow["id"]
                pname = prow["name"]
                pcolor = (
                    prow["color"]
                    if isinstance(prow["color"], str) and prow["color"]
                    else "#3788d8"
                )
                ptasks = all_tasks[all_tasks["part_id"] == pid]
                val = completion_ratio(ptasks) if not ptasks.empty else 0
                items.append(
                    {
                        "label": pname,
                        "value": val,
                        "color": pcolor,
                    }
                )

            n_items = len(items)
            max_cols = 4
            idx = 0
            while idx < n_items:
                cols = st.columns(min(max_cols, n_items - idx))
                for c in range(len(cols)):
                    item = items[idx]
                    with cols[c]:
                        CircularProgress(
                            label=item["label"],
                            value=item["value"],
                            key=f"cp_{item['label']}_{idx}",
                            color=item["color"],
                        ).st_circular_progress()
                    idx += 1


# =========================================================
# Fragment: 파트별 보드
# =========================================================
@st.fragment()
def render_part_board(
    part_name,
    selected_project_id,
    parts_df,
):
    st.subheader(f"🗂 {part_name} 파트 작업 보드")

    if not selected_project_id:
        st.info("좌측에서 프로젝트를 선택하세요.")
        return

    part_row = parts_df[parts_df["name"] == part_name]
    if part_row.empty:
        st.error("해당 파트 정보를 찾을 수 없습니다.")
        return

    part_id = int(part_row["id"].iloc[0])
    tdf = list_tasks(project_id=selected_project_id, part_id=part_id)

    # --- 날짜 상태 기본값: 오늘 ---
    key_sel = f"part_{part_id}_selected_date"
    if key_sel not in st.session_state:
        st.session_state[key_sel] = date.today().isoformat()
    current_sel_str = st.session_state[key_sel]

    events = build_calendar_events(tdf, show_part_in_title=False)
    options = calendar_options_base()
    options["initialDate"] = current_sel_str

    cal_val = st_calendar(
        events=events,
        options=options,
        key=f"calendar_part_{part_id}",
    )

    if isinstance(cal_val, dict) and cal_val.get("callback") == "dateClick":
        dc = cal_val.get("dateClick", {})
        clicked = normalize_clicked_date(dc)
        if clicked:
            st.session_state[key_sel] = clicked
            current_sel_str = clicked

    selected_day = date.fromisoformat(current_sel_str)

    with st.expander("🔍 필터", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            assignee_filter = st.text_input("담당자(부분일치)")
        with f2:
            status_filter = st.multiselect(
                "상태", ["Todo", "In Progress", "Done"]
            )
        with f3:
            priority_filter = st.multiselect(
                "우선순위", ["Low", "Medium", "High"]
            )
        with f4:
            tag_filter = st.text_input("태그(부분일치)")

        def apply_filters(df):
            if df.empty:
                return df
            res = df.copy()
            if assignee_filter:
                res = res[
                    res["assignee"]
                    .fillna("")
                    .str.contains(assignee_filter, case=False)
                ]
            if status_filter:
                res = res[res["status"].isin(status_filter)]
            if priority_filter:
                res = res[res["priority"].isin(priority_filter)]
            if tag_filter:
                res = res[
                    res["tags"]
                    .fillna("")
                    .str.contains(tag_filter, case=False)
                ]
            return res

    tdf_f = apply_filters(tdf) if not tdf.empty else tdf

    part_users_df = get_users_for_part(part_id)
    if not part_users_df.empty:
        user_options = ["(없음)"] + part_users_df["name"].tolist()
    else:
        user_options = ["(없음)"]

    col_todo, col_prog, col_done = st.columns(3)

    for label, col in [
        ("Todo", col_todo),
        ("In Progress", col_prog),
        ("Done", col_done),
    ]:
        with col:
            st.markdown(f"### {label}")
            df_col = tdf_f[tdf_f["status"] == label]
            if df_col.empty:
                st.caption("비어 있음")
            else:
                for _, r in df_col.iterrows():
                    task_id = int(r["id"])
                    edit_key = f"edit_mode_{task_id}"
                    edit_mode = st.session_state.get(edit_key, False)

                    with st.container(border=True):
                        priority = r["priority"] or "Medium"
                        pr_label, pr_color = priority_label_and_color(priority)

                        if not edit_mode:
                            # 보기 모드
                            st.markdown(
                                f"""
                                <div style="display:flex;align-items:center;gap:8px;">
                                  <span style="font-weight:600;">{r['title']}</span>
                                  <span style="font-size:0.8rem;padding:2px 8px;border-radius:999px;
                                               background-color:{pr_color};color:#000;">
                                    {pr_label}
                                  </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # 서브태스크 체크 → 상태/진행률 자동 반영
                            subtasks_orig = parse_subtasks(
                                r.get("description") or ""
                            )
                            if subtasks_orig:
                                task_status = r.get("status") or "Todo"
                                # Done이면 화면에서도 모두 체크
                                if task_status == "Done":
                                    subtasks_for_view = [
                                        (lbl, weight, True)
                                        for (lbl, weight, done) in subtasks_orig
                                    ]
                                else:
                                    subtasks_for_view = subtasks_orig[:]

                                changed = False
                                new_subtasks = []
                                for i, (
                                    lbl,
                                    weight,
                                    done_flag,
                                ) in enumerate(subtasks_for_view):
                                    key_cb = f"subtask_cb_{task_id}_{i}"
                                    checked = st.checkbox(
                                        f"{lbl} ({weight}%)",
                                        value=done_flag,
                                        key=key_cb,
                                    )
                                    if checked != done_flag:
                                        changed = True
                                    new_subtasks.append((lbl, weight, checked))

                                if changed:
                                    new_desc = serialize_subtasks(new_subtasks)
                                    new_prog = calc_progress_from_subtasks(
                                        new_subtasks
                                    )

                                    if new_prog == 0:
                                        new_status = "Todo"
                                    elif new_prog == 100:
                                        new_status = "Done"
                                    else:
                                        new_status = "In Progress"

                                    update_task(
                                        task_id,
                                        description=new_desc,
                                        progress=int(new_prog),
                                        status=new_status,
                                    )
                                    st.rerun()

                            # 진행률/담당/마감 표시
                            st.caption(
                                f"담당: {r['assignee'] or '-'} · "
                                f"마감: {r['due_date'] or '-'} · 진행률: {r.get('progress') or 0}%"
                            )

                            b_done, b_edit, b_del = st.columns(3, gap="small")
                            with b_done:
                                if st.button(
                                    "완료",
                                    key=f"done_btn_{task_id}",
                                    use_container_width=True,
                                ):
                                    subtasks_all = parse_subtasks(
                                        r.get("description") or ""
                                    )
                                    if subtasks_all:
                                        new_subtasks_all = [
                                            (lbl, w, True)
                                            for (lbl, w, d) in subtasks_all
                                        ]
                                        new_desc = serialize_subtasks(
                                            new_subtasks_all
                                        )
                                    else:
                                        new_desc = r.get("description") or None
                                    update_task(
                                        task_id,
                                        status="Done",
                                        progress=100,
                                        description=new_desc,
                                    )
                                    st.rerun()
                            with b_edit:
                                if st.button(
                                    "수정",
                                    key=f"edit_btn_{task_id}",
                                    use_container_width=True,
                                ):
                                    st.session_state[edit_key] = True
                                    st.rerun()
                            with b_del:
                                if st.button(
                                    "삭제",
                                    key=f"del_{task_id}",
                                    use_container_width=True,
                                ):
                                    st.session_state[
                                        f"confirm_del_task_{task_id}"
                                    ] = True

                            if st.session_state.get(
                                f"confirm_del_task_{task_id}"
                            ):
                                st.warning(
                                    "정말 삭제할까요? 아래 버튼을 누르면 삭제됩니다."
                                )
                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    if st.button(
                                        "네, 삭제합니다",
                                        key=f"confirm_del_task_btn_{task_id}",
                                        use_container_width=True,
                                    ):
                                        delete_task(task_id)
                                        st.session_state.pop(
                                            f"confirm_del_task_{task_id}", None
                                        )
                                        st.warning("작업이 삭제되었습니다.")
                                        st.rerun()
                                with c2:
                                    if st.button(
                                        "취소",
                                        key=f"cancel_del_task_{task_id}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.pop(
                                            f"confirm_del_task_{task_id}", None
                                        )

                        else:
                            # 수정 모드
                            st.markdown("**수정 모드**")
                            title_val = st.text_input(
                                "제목",
                                value=r["title"],
                                key=f"edit_title_{task_id}",
                            )

                            assignee_current = r["assignee"] or "(없음)"
                            assignee_val = st.selectbox(
                                "담당자",
                                user_options,
                                index=user_options.index(assignee_current)
                                if assignee_current in user_options
                                else 0,
                                key=f"edit_assignee_{task_id}",
                            )

                            # 🔹 시작일 / 마감일 수정 가능
                            def _parse_date(v):
                                if isinstance(v, str) and v:
                                    try:
                                        return date.fromisoformat(v)
                                    except Exception:
                                        pass
                                return date.today()

                            c_d1, c_d2 = st.columns(2)
                            with c_d1:
                                start_date_val = st.date_input(
                                    "시작일",
                                    value=_parse_date(r.get("start_date")),
                                    key=f"edit_start_{task_id}",
                                )
                            with c_d2:
                                due_date_val = st.date_input(
                                    "마감일",
                                    value=_parse_date(r.get("due_date")),
                                    key=f"edit_due_{task_id}",
                                )

                            # 🔹 서브태스크 편집
                            subtasks = parse_subtasks(
                                r.get("description") or ""
                            )
                            n_rows = max(len(subtasks), 1)
                            edit_subtasks = []

                            for i in range(n_rows):
                                if i < len(subtasks):
                                    d_label, d_weight, d_done = subtasks[i]
                                else:
                                    d_label, d_weight, d_done = "", 0, False
                                c_l, c_p = st.columns([4, 1])
                                with c_l:
                                    lbl = st.text_input(
                                        f"세부 작업 {i+1}",
                                        value=d_label,
                                        key=f"edit_sub_label_{task_id}_{i}",
                                    )
                                with c_p:
                                    weight_val = st.number_input(
                                        "할당률 (%)",
                                        min_value=0,
                                        max_value=100,
                                        value=int(d_weight),
                                        key=f"edit_sub_prog_{task_id}_{i}",
                                    )
                                if lbl.strip():
                                    edit_subtasks.append(
                                        (lbl.strip(), weight_val, d_done)
                                    )

                            tags_val = st.text_input(
                                "태그(쉼표 구분)",
                                value=r.get("tags") or "",
                                key=f"edit_tags_{task_id}",
                            )

                            b1, b2 = st.columns(2, gap="small")
                            with b1:
                                if st.button(
                                    "저장",
                                    key=f"save_edit_{task_id}",
                                    use_container_width=True,
                                ):
                                    if edit_subtasks:
                                        new_desc = serialize_subtasks(
                                            edit_subtasks
                                        )
                                        new_prog = (
                                            calc_progress_from_subtasks(
                                                edit_subtasks
                                            )
                                        )
                                    else:
                                        new_desc = None
                                        new_prog = 0

                                    if new_prog == 0:
                                        new_status = "Todo"
                                    elif new_prog == 100:
                                        new_status = "Done"
                                    else:
                                        new_status = "In Progress"

                                    assignee_final = (
                                        None
                                        if assignee_val == "(없음)"
                                        else assignee_val
                                    )

                                    update_task(
                                        task_id,
                                        title=title_val.strip()
                                        or r["title"],
                                        status=new_status,
                                        description=new_desc,
                                        progress=int(new_prog),
                                        assignee=assignee_final,
                                        tags=tags_val.strip() or None,
                                        start_date=start_date_val.isoformat()
                                        if start_date_val
                                        else None,
                                        due_date=due_date_val.isoformat()
                                        if due_date_val
                                        else None,
                                    )
                                    st.session_state[edit_key] = False
                                    st.success("수정되었습니다.")
                                    st.rerun()
                            with b2:
                                if st.button(
                                    "취소",
                                    key=f"cancel_edit_{task_id}",
                                    use_container_width=True,
                                ):
                                    st.session_state[edit_key] = False
                                    st.rerun()

    # 새 작업 추가
    st.divider()
    st.markdown("### ➕ 새 작업 추가")

    count_key = f"subtask_count_{part_id}"
    if count_key not in st.session_state:
        st.session_state[count_key] = 1

    with st.form(f"add_task_{part_id}"):

        c_title, c_tag = st.columns([2, 1])
        with c_title:
            title = st.text_input(
                "제목*",
                placeholder="예: API 연동 구현",
                key=f"title_input_{part_id}",
            )
        with c_tag:
            tags = st.text_input(
                "태그(쉼표 구분)",
                placeholder="백엔드,UI 등",
                key=f"tag_input_{part_id}",
            )

        c1, c2 = st.columns(2)
        with c1:
            assignee_choice = st.selectbox(
                "담당자", user_options, key=f"assignee_{part_id}"
            )
        with c2:
            status = st.selectbox(
                "상태",
                ["Todo", "In Progress", "Done"],
                key=f"status_new_{part_id}",
            )

        c3, c4 = st.columns(2)
        with c3:
            start_date = st.date_input(
                "시작일",
                value=selected_day,
                key=f"start_{part_id}",
            )
        with c4:
            due_date = st.date_input(
                "마감일",
                value=selected_day,
                key=f"due_{part_id}",
            )

        sub_labels = []
        sub_weights = []
        for i in range(st.session_state[count_key]):
            c_l, c_p = st.columns([3, 1])
            with c_l:
                lbl = st.text_input(
                    f"세부 작업 {i+1}",
                    key=f"new_sub_label_{part_id}_{i}",
                )
            with c_p:
                prog_val = st.number_input(
                    "할당률 (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    key=f"new_sub_prog_{part_id}_{i}",
                )
            if lbl.strip():
                sub_labels.append(lbl.strip())
                sub_weights.append(prog_val)

        b1, b2 = st.columns(2, gap="small")
        add_clicked = b1.form_submit_button(
            "세부 작업 추가", use_container_width=True
        )
        save_clicked = b2.form_submit_button(
            "저장", use_container_width=True
        )

        if add_clicked:
            st.session_state[count_key] += 1

        if save_clicked:
            if not title.strip():
                st.error("제목은 필수입니다.")
            else:
                if assignee_choice == "(없음)":
                    assignee_val = None
                else:
                    assignee_val = assignee_choice

                subtasks_new = []
                for lbl, w in zip(sub_labels, sub_weights):
                    done_flag = True if status == "Done" else False
                    subtasks_new.append((lbl, w, done_flag))

                if subtasks_new:
                    description_str = serialize_subtasks(subtasks_new)
                else:
                    description_str = None

                if status == "Done":
                    progress = 100
                else:
                    progress = 0

                insert_task(
                    project_id=selected_project_id,
                    part_id=part_id,
                    title=title.strip(),
                    description=description_str,
                    assignee=assignee_val,
                    priority="Medium",
                    status=status,
                    start_date=start_date.isoformat()
                    if start_date
                    else None,
                    due_date=due_date.isoformat()
                    if due_date
                    else None,
                    progress=int(progress),
                    tags=tags.strip() or None,
                )
                st.success("작업이 추가되었습니다.")
                st.rerun()


# =========================================================
# Streamlit 설정 및 로그인
# =========================================================
st.set_page_config(page_title="협업툴 - 일정/진행도", layout="wide")

st.markdown(
    """
<style>
.red-button button {
    background-color: #ff4b4b !important;
    color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- DB 초기화/시드 (캐시된 함수 호출) ---
ensure_db_initialized()

auth_cfg = st.secrets.get("auth", {})
COMPANY_NAME = auth_cfg.get("company_name", "Inha")
ADMIN_USERNAME = auth_cfg.get("admin_username", "admin")
ADMIN_PASSWORD = auth_cfg.get("admin_password", "1234")
USER_USERNAME = auth_cfg.get("user_username", "user")
USER_PASSWORD = auth_cfg.get("user_password", "1234")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["role"] = None
    st.session_state["current_tab"] = "대시보드"

# 로그인
if not st.session_state["logged_in"]:
    st.title("협업툴 로그인 (ID: admin, PW: 1234")

    with st.form("login_form"):
        company = st.selectbox("회사", [COMPANY_NAME], index=0)
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        login_btn = st.form_submit_button("로그인")

        if login_btn:
            ok = False
            role = None

            if company == COMPANY_NAME:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    ok = True
                    role = "admin"
                elif username == USER_USERNAME and password == USER_PASSWORD:
                    ok = True
                    role = "user"

            if ok:
                st.session_state["logged_in"] = True
                st.session_state["role"] = role
                st.session_state["current_tab"] = "대시보드"
                st.rerun()
            else:
                st.error("로그인 정보가 올바르지 않습니다.")
    st.stop()

CURRENT_USER = "기획자 A"  # 데모용


# =========================================================
# 공용 데이터
# =========================================================
projects_df = list_projects()
parts_df = list_parts()
part_names = parts_df["name"].tolist()
users_df = list_users()

if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = "대시보드"

if st.session_state["role"] == "user" and st.session_state["current_tab"] in [
    "프로젝트 관리",
    "유저 관리",
]:
    st.session_state["current_tab"] = "대시보드"

# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.markdown("### 프로젝트")
    if projects_df.empty:
        selected_project_id = None
        selected_project_name = ""
        st.selectbox("", ["프로젝트 없음"], disabled=True)
    else:
        project_names = projects_df["name"].tolist()
        selected_project_name = st.selectbox("", project_names)
        selected_project_id = int(
            projects_df[projects_df["name"] == selected_project_name]["id"].iloc[0]
        )

    st.write("")
    if st.button("대시보드", use_container_width=True):
        st.session_state["current_tab"] = "대시보드"

    st.markdown("---")
    st.write("### 파트")

    for pname in part_names:
        if st.button(pname, use_container_width=True, key=f"tab_{pname}"):
            st.session_state["current_tab"] = f"파트:{pname}"

    if st.session_state["role"] == "admin":
        st.markdown("---")
        st.markdown("### 관리자")
        if st.button("프로젝트 관리", use_container_width=True):
            st.session_state["current_tab"] = "프로젝트 관리"
        if st.button("유저 관리", use_container_width=True):
            st.session_state["current_tab"] = "유저 관리"

    st.markdown("---")
    if st.button("로그아웃", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["role"] = None
        st.session_state["current_tab"] = "대시보드"
        st.rerun()

current_tab = st.session_state["current_tab"]

if selected_project_id:
    st.title(selected_project_name)
else:
    st.title("프로젝트가 없습니다")


# =========================================================
# 대시보드
# =========================================================
if current_tab == "대시보드":
    render_dashboard(
        selected_project_id,
        parts_df,
        part_names,
        CURRENT_USER,
        st.session_state["role"],
    )

# =========================================================
# 프로젝트 관리 (admin)
# =========================================================
elif current_tab == "프로젝트 관리" and st.session_state["role"] == "admin":
    st.subheader("🧩 프로젝트 관리")

    top_left, top_right = st.columns(2)

    # ----------------------------
    # 프로젝트 목록 / 수정 / 삭제
    # ----------------------------
    with top_left:
        st.markdown("#### 프로젝트 목록 / 수정")

        projects_df = list_projects()
        if projects_df.empty:
            st.caption("등록된 프로젝트가 없습니다.")
        else:
            for _, row in projects_df.iterrows():
                pid = int(row["id"])

                # 이름 + 버튼 (설명 입력 칸 제거, 버튼 영역은 파트랑 동일하게 1:1)
                c1, c2 = st.columns([4, 2])

                with c1:
                    new_name = st.text_input(
                        "프로젝트 이름",
                        value=row["name"],
                        key=f"proj_name_{pid}",
                        label_visibility="collapsed",
                    )

                with c2:
                    b1, b2 = st.columns(2)  # 파트와 동일 구조
                    with b1:
                        if st.button(
                            "저장",
                            key=f"proj_save_{pid}",
                            use_container_width=True,
                        ):
                            # 설명은 여기서 안 바꾸고, 이름만 수정
                            update_project(
                                pid,
                                name=new_name.strip() or row["name"],
                            )
                            st.success("프로젝트가 수정되었습니다.")
                            st.rerun()
                    with b2:
                        if st.button(
                            "삭제",
                            key=f"proj_del_{pid}",
                            use_container_width=True,
                        ):
                            st.session_state["confirm_del_project"] = pid

        # 프로젝트 삭제 확인
        if st.session_state.get("confirm_del_project") is not None:
            pid = st.session_state["confirm_del_project"]
            st.warning(
                f"정말 프로젝트(id={pid})를 삭제할까요? 관련 작업/권한도 함께 삭제됩니다."
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "네, 삭제합니다",
                    key="proj_del_confirm_btn",
                    use_container_width=True,
                ):
                    delete_project(pid)
                    st.session_state.pop("confirm_del_project", None)
                    st.success("프로젝트가 삭제되었습니다.")
                    st.rerun()
            with c2:
                if st.button(
                    "취소",
                    key="proj_del_cancel_btn",
                    use_container_width=True,
                ):
                    st.session_state.pop("confirm_del_project", None)

    # ----------------------------
    # 파트 목록 / 수정 / 삭제
    # ----------------------------
    with top_right:
        st.markdown("#### 파트 목록 / 수정")
        parts_df = list_parts()
        if parts_df.empty:
            st.caption("등록된 파트가 없습니다.")
        else:
            for _, row in parts_df.iterrows():
                pid = int(row["id"])
                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    new_part_name = st.text_input(
                        "이름",
                        value=row["name"],
                        key=f"part_name_{pid}",
                        label_visibility="collapsed",
                    )
                with c2:
                    current_color = (
                        row["color"]
                        if isinstance(row["color"], str) and row["color"]
                        else "#3788d8"
                    )
                    color_val = st.color_picker(
                        "색상",
                        current_color,
                        key=f"part_color_{pid}",
                        label_visibility="collapsed",
                    )
                with c3:
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button(
                            "저장",
                            key=f"save_part_{pid}",
                            use_container_width=True,
                        ):
                            update_part(
                                pid,
                                name=new_part_name.strip() or row["name"],
                                color=color_val,
                            )
                            st.success(f"{row['name']} 파트가 업데이트되었습니다.")
                            st.rerun()
                    with b2:
                        if st.button(
                            "삭제",
                            key=f"del_part_{pid}",
                            use_container_width=True,
                        ):
                            st.session_state["confirm_del_part"] = pid

        # 파트 삭제 확인
        if st.session_state.get("confirm_del_part") is not None:
            pid = st.session_state["confirm_del_part"]
            st.warning(
                f"정말 파트(id={pid})를 삭제할까요? 관련 작업/유저-파트 연결도 함께 삭제됩니다."
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "네, 삭제합니다",
                    key="part_del_confirm_btn",
                    use_container_width=True,
                ):
                    delete_part(pid)
                    st.session_state.pop("confirm_del_part", None)
                    st.success("파트가 삭제되었습니다.")
                    st.rerun()
            with c2:
                if st.button(
                    "취소",
                    key="part_del_cancel_btn",
                    use_container_width=True,
                ):
                    st.session_state.pop("confirm_del_part", None)

    st.markdown("---")

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.markdown("#### 프로젝트 추가")
        with st.form("add_project"):
            p_name = st.text_input("프로젝트 이름*", key="new_proj_name")
            p_desc = st.text_input("설명", key="new_proj_desc")
            add_proj = st.form_submit_button("추가")
            if add_proj:
                if not p_name.strip():
                    st.error("프로젝트 이름은 필수입니다.")
                else:
                    insert_project(p_name.strip(), p_desc.strip())
                    st.success("프로젝트가 추가되었습니다.")
                    st.rerun()

    with bottom_right:
        st.markdown("#### 파트 추가")
        with st.form("add_part"):
            new_part_name = st.text_input(
                "새 파트 이름", placeholder="예: QA, 운영 등", key="new_part_name"
            )
            new_part_color = st.color_picker(
                "색상", "#3788d8", key="new_part_color"
            )
            submitted = st.form_submit_button("추가")
            if submitted:
                parts_df = list_parts()
                if not new_part_name.strip():
                    st.error("파트명을 입력하세요.")
                elif new_part_name.strip() in parts_df["name"].tolist():
                    st.error("이미 존재하는 파트입니다.")
                else:
                    insert_part(new_part_name.strip(), new_part_color)
                    st.success("파트가 추가되었습니다.")
                    st.rerun()

# =========================================================
# 유저 관리 (admin)
# =========================================================
elif current_tab == "유저 관리" and st.session_state["role"] == "admin":
    st.subheader("👤 유저 관리")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 유저 목록")
        users_df = list_users()
        if users_df.empty:
            st.info("등록된 유저가 없습니다.")
        else:
            show_cols = ["name", "email", "part_names", "role"]
            exist_cols = [c for c in show_cols if c in users_df.columns]
            st.dataframe(
                users_df[exist_cols], use_container_width=True, hide_index=True
            )

    with col2:
        st.markdown("#### 유저 추가")
        with st.form("add_user"):
            u_name = st.text_input("이름*")
            u_email = st.text_input("이메일")
            u_role = st.text_input("역할", placeholder="예: planner, dev 등")
            parts_selected = st.multiselect("파트(여러 개 선택 가능)", part_names)
            submitted = st.form_submit_button("유저 생성")
            if submitted:
                if not u_name.strip():
                    st.error("이름은 필수입니다.")
                else:
                    part_ids = []
                    parts_df = list_parts()
                    for pn in parts_selected:
                        pid = int(
                            parts_df[parts_df["name"] == pn]["id"].iloc[0]
                        )
                        part_ids.append(pid)
                    insert_user(
                        u_name.strip(),
                        u_email.strip() or None,
                        part_ids,
                        u_role.strip() or None,
                    )
                    st.success("유저가 추가되었습니다.")
                    st.rerun()

        st.write("")
        users_df = list_users()
        if users_df.empty:
            st.info("유저가 없습니다.")
        else:
            st.markdown("#### 유저 상세 설정")
            with st.container(border=True):
                user_labels = [
                    f"{r['name']} ({r['email'] or '-'})"
                    for _, r in users_df.iterrows()
                ]
                selected_label = st.selectbox(
                    "유저 선택",
                    user_labels,
                    key="user_select",
                )
                idx = user_labels.index(selected_label)
                user_row = users_df.iloc[idx]
                user_id = int(user_row["id"])

                parts_df = list_parts()
                projects_df = list_projects()
                proj_names = projects_df["name"].tolist()
                proj_id_by_name = {
                    r["name"]: int(r["id"]) for _, r in projects_df.iterrows()
                }

                user_parts_df = get_parts_for_user(user_id)
                current_part_names = (
                    user_parts_df["name"].tolist()
                    if not user_parts_df.empty
                    else []
                )
                user_proj_df = get_projects_for_user(user_id)
                current_proj_ids = (
                    user_proj_df["id"].tolist()
                    if not user_proj_df.empty
                    else []
                )
                current_proj_names = [
                    name
                    for name in proj_names
                    if proj_id_by_name[name] in current_proj_ids
                ]

                new_parts = st.multiselect(
                    "파트",
                    part_names,
                    default=current_part_names,
                )
                new_proj_names = st.multiselect(
                    "접속 가능한 프로젝트",
                    proj_names,
                    default=current_proj_names,
                )

                btn_col1, btn_col2 = st.columns(2, gap="small")
                with btn_col1:
                    if st.button(
                        "저장 및 수정",
                        key="save_user_parts",
                        use_container_width=True,
                    ):
                        new_part_ids = []
                        for pn in new_parts:
                            pid = int(
                                parts_df[parts_df["name"] == pn]["id"].iloc[0]
                            )
                            new_part_ids.append(pid)
                        set_user_parts(user_id, new_part_ids)
                        main_part_id = new_part_ids[0] if new_part_ids else None
                        update_user(user_id, part_id=main_part_id)

                        new_proj_ids = [proj_id_by_name[n] for n in new_proj_names]
                        set_user_projects(user_id, new_proj_ids)
                        st.success("설정이 저장·수정되었습니다.")
                        st.rerun()

                with btn_col2:
                    del_clicked = st.button(
                        "유저 삭제",
                        key=f"del_user_{user_id}",
                        use_container_width=True,
                    )
                    if del_clicked:
                        st.session_state["confirm_del_user"] = user_id

        if (
            "confirm_del_user" in st.session_state
            and not users_df.empty
        ):
            cid = st.session_state.get("confirm_del_user")
            if cid is not None:
                st.warning("정말 삭제할까요? 아래 버튼을 누르면 삭제됩니다.")
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button(
                        "네, 삭제합니다",
                        key=f"confirm_del_user_btn_{cid}",
                        use_container_width=True,
                    ):
                        delete_user(cid)
                        st.session_state.pop("confirm_del_user", None)
                        st.warning("유저가 삭제되었습니다.")
                        st.rerun()
                with c2:
                    if st.button(
                        "취소",
                        key=f"cancel_del_user_{cid}",
                        use_container_width=True,
                    ):
                        st.session_state.pop("confirm_del_user", None)

# =========================================================
# 파트별 화면
# =========================================================
else:
    if current_tab.startswith("파트:"):
        part_name = current_tab.split("파트:", 1)[1]
    else:
        part_name = current_tab

    st.subheader(f"🗂 {part_name} 파트 작업 보드")

    if not selected_project_id:
        st.info("좌측에서 프로젝트를 선택하세요.")
    else:
        parts_df = list_parts()
        part_row = parts_df[parts_df["name"] == part_name]
        if part_row.empty:
            st.error("해당 파트 정보를 찾을 수 없습니다.")
        else:
            part_id = int(part_row["id"].iloc[0])
            tdf = list_tasks(project_id=selected_project_id, part_id=part_id)

            events = build_calendar_events(tdf, show_part_in_title=False)
            options = calendar_options_base()
            cal_val = st_calendar(
                events=events,
                options=options,
                key=f"calendar_part_{part_id}",
            )

            key_sel = f"part_{part_id}_selected_date"
            default_sel = st.session_state.get(key_sel, date.today().isoformat())
            if isinstance(cal_val, dict) and cal_val.get("callback") == "dateClick":
                d_str = cal_val["dateClick"]["date"][:10]
                st.session_state[key_sel] = d_str
                default_sel = d_str
            selected_day = date.fromisoformat(default_sel)

            with st.expander("🔍 필터", expanded=False):
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    assignee_filter = st.text_input("담당자(부분일치)")
                with f2:
                    status_filter = st.multiselect(
                        "상태", ["Todo", "In Progress", "Done"]
                    )
                with f3:
                    priority_filter = st.multiselect(
                        "우선순위", ["Low", "Medium", "High"]
                    )
                with f4:
                    tag_filter = st.text_input("태그(부분일치)")

                def apply_filters(df):
                    if df.empty:
                        return df
                    res = df.copy()
                    if assignee_filter:
                        res = res[
                            res["assignee"]
                            .fillna("")
                            .str.contains(assignee_filter, case=False)
                        ]
                    if status_filter:
                        res = res[res["status"].isin(status_filter)]
                    if priority_filter:
                        res = res[res["priority"].isin(priority_filter)]
                    if tag_filter:
                        res = res[
                            res["tags"]
                            .fillna("")
                            .str.contains(tag_filter, case=False)
                        ]
                    return res

            tdf_f = apply_filters(tdf) if not tdf.empty else tdf

            part_users_df = get_users_for_part(part_id)
            if not part_users_df.empty:
                user_options = ["(없음)"] + part_users_df["name"].tolist()
            else:
                user_options = ["(없음)"]

            col_todo, col_prog, col_done = st.columns(3)

            priority_options = ["Low", "Medium", "High"]
            priority_labels = {"Low": "낮음", "Medium": "보통", "High": "높음"}

            for label, col in [
                ("Todo", col_todo),
                ("In Progress", col_prog),
                ("Done", col_done),
            ]:
                with col:
                    st.markdown(f"### {label}")
                    df_col = tdf_f[tdf_f["status"] == label]
                    if df_col.empty:
                        st.caption("비어 있음")
                    else:
                        for _, r in df_col.iterrows():
                            task_id = int(r["id"])
                            edit_key = f"edit_mode_{task_id}"
                            edit_mode = st.session_state.get(edit_key, False)

                            with st.container(border=True):
                                priority = r["priority"] or "Medium"
                                pr_label, pr_color = priority_label_and_color(
                                    priority
                                )

                                if not edit_mode:
                                    # 보기 모드
                                    st.markdown(
                                        f"""
                                        <div style="display:flex;align-items:center;gap:8px;">
                                          <span style="font-weight:600;">{r['title']}</span>
                                          <span style="font-size:0.8rem;padding:2px 8px;border-radius:999px;
                                                       background-color:{pr_color};color:#000;">
                                            {pr_label}
                                          </span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    # ---------------------------
                                    # 서브태스크 체크 → 상태/진행률 자동 반영
                                    # ---------------------------
                                    subtasks_orig = parse_subtasks(r.get("description") or "")
                                    changed = False

                                    # 화면에 보여줄 진행률/상태 (기본은 DB 값)
                                    task_progress = int(r.get("progress") or 0)
                                    task_status = r.get("status") or "Todo"

                                    if subtasks_orig:
                                        # Done이면 화면에서도 모두 체크 처리
                                        if task_status == "Done":
                                            subtasks_for_view = [
                                                (lbl, weight, True)
                                                for (lbl, weight, done) in subtasks_orig
                                            ]
                                        else:
                                            subtasks_for_view = subtasks_orig[:]

                                        new_subtasks = []
                                        for i, (lbl, weight, done_flag) in enumerate(subtasks_for_view):
                                            key_cb = f"view_sub_done_{task_id}_{i}"
                                            checked = st.checkbox(
                                                f"{lbl} ({weight}%)",
                                                value=done_flag,
                                                key=key_cb,
                                            )
                                            if checked != done_flag:
                                                changed = True
                                            new_subtasks.append((lbl, weight, checked))

                                        if changed:
                                            # 새 진행률/상태 계산
                                            new_desc = serialize_subtasks(new_subtasks)
                                            new_prog = calc_progress_from_subtasks(new_subtasks)

                                            if new_prog == 0:
                                                new_status = "Todo"
                                            elif new_prog == 100:
                                                new_status = "Done"
                                            else:
                                                new_status = "In Progress"

                                            # DB 즉시 업데이트
                                            update_task(
                                                task_id,
                                                description=new_desc,
                                                progress=int(new_prog),
                                                status=new_status,
                                            )

                                            # 다음 런에서 칸반 위치까지 바로 반영
                                            st.rerun()

                                    # 진행률/담당/마감 표시 (서브태스크 없어도 항상 표시)
                                    st.caption(
                                        f"담당: {r['assignee'] or '-'} · "
                                        f"마감: {r['due_date'] or '-'} · 진행률: {task_progress}%"
                                    )

                                    b_done, b_edit, b_del = st.columns(
                                        3, gap="small"
                                    )
                                    with b_done:
                                        if st.button(
                                            "완료",
                                            key=f"done_btn_{task_id}",
                                            use_container_width=True,
                                        ):
                                            subtasks_all = parse_subtasks(
                                                r.get("description") or ""
                                            )
                                            if subtasks_all:
                                                new_subtasks_all = [
                                                    (lbl, w, True)
                                                    for (lbl, w, d) in subtasks_all
                                                ]
                                                new_desc = serialize_subtasks(
                                                    new_subtasks_all
                                                )
                                            else:
                                                new_desc = (
                                                    r.get("description") or None
                                                )
                                            update_task(
                                                task_id,
                                                status="Done",
                                                progress=100,
                                                description=new_desc,
                                            )
                                            st.rerun()
                                    with b_edit:
                                        if st.button(
                                            "수정",
                                            key=f"edit_btn_{task_id}",
                                            use_container_width=True,
                                        ):
                                            st.session_state[edit_key] = True
                                            st.rerun()
                                    with b_del:
                                        if st.button(
                                            "삭제",
                                            key=f"del_{task_id}",
                                            use_container_width=True,
                                        ):
                                            st.session_state[
                                                f"confirm_del_task_{task_id}"
                                            ] = True

                                    if st.session_state.get(
                                        f"confirm_del_task_{task_id}"
                                    ):
                                        st.warning(
                                            "정말 삭제할까요? 아래 버튼을 누르면 삭제됩니다."
                                        )
                                        c1, c2 = st.columns([1, 1])
                                        with c1:
                                            if st.button(
                                                "네, 삭제합니다",
                                                key=f"confirm_del_task_btn_{task_id}",
                                                use_container_width=True,
                                            ):
                                                delete_task(task_id)
                                                st.session_state.pop(
                                                    f"confirm_del_task_{task_id}",
                                                    None,
                                                )
                                                st.warning("작업이 삭제되었습니다.")
                                                st.rerun()
                                        with c2:
                                            if st.button(
                                                "취소",
                                                key=f"cancel_del_task_{task_id}",
                                                use_container_width=True,
                                            ):
                                                st.session_state.pop(
                                                    f"confirm_del_task_{task_id}",
                                                    None,
                                                )

                                else:
                                    # 수정 모드
                                    st.markdown("**수정 모드**")

                                    # 제목
                                    title_val = st.text_input(
                                        "제목",
                                        value=r["title"],
                                        key=f"edit_title_{task_id}",
                                    )

                                    # 담당자
                                    assignee_current = r["assignee"] or "(없음)"
                                    assignee_val = st.selectbox(
                                        "담당자",
                                        user_options,
                                        index=user_options.index(assignee_current)
                                        if assignee_current in user_options
                                        else 0,
                                        key=f"edit_assignee_{task_id}",
                                    )

                                    # 🔹 날짜 파싱 헬퍼
                                    def _parse_date(v):
                                        if isinstance(v, str) and v:
                                            try:
                                                return date.fromisoformat(v[:10])
                                            except Exception:
                                                pass
                                        try:
                                            if pd.notna(v):
                                                return v.date()
                                        except Exception:
                                            pass
                                        return date.today()

                                    # 🔹 시작일 / 마감일 수정
                                    c_d1, c_d2 = st.columns(2)
                                    with c_d1:
                                        start_date_val = st.date_input(
                                            "시작일",
                                            value=_parse_date(r.get("start_date")),
                                            key=f"edit_start_{task_id}",
                                        )
                                    with c_d2:
                                        due_date_val = st.date_input(
                                            "마감일",
                                            value=_parse_date(r.get("due_date")),
                                            key=f"edit_due_{task_id}",
                                        )

                                    # 🔹 태그 + 우선순위 한 줄 반반
                                    c_tag, c_pri = st.columns(2)
                                    with c_tag:
                                        tags_val = st.text_input(
                                            "태그(쉼표 구분)",
                                            value=r.get("tags") or "",
                                            key=f"edit_tags_{task_id}",
                                        )
                                    with c_pri:
                                        current_priority = r["priority"] or "Medium"
                                        if current_priority not in priority_options:
                                            current_priority = "Medium"
                                        priority_val = st.selectbox(
                                            "우선순위",
                                            priority_options,
                                            index=priority_options.index(current_priority),
                                            format_func=lambda v: priority_labels.get(v, v),
                                            key=f"edit_priority_{task_id}",
                                        )

                                    # 🔹 서브태스크 편집
                                    subtasks = parse_subtasks(
                                        r.get("description") or ""
                                    )
                                    n_rows = max(len(subtasks), 1)
                                    edit_subtasks = []

                                    for i in range(n_rows):
                                        if i < len(subtasks):
                                            d_label, d_weight, d_done = subtasks[i]
                                        else:
                                            d_label, d_weight, d_done = "", 0, False
                                        c_l, c_p = st.columns([4, 1])
                                        with c_l:
                                            lbl = st.text_input(
                                                f"세부 작업 {i+1}",
                                                value=d_label,
                                                key=f"edit_sub_label_{task_id}_{i}",
                                            )
                                        with c_p:
                                            weight_val = st.number_input(
                                                "할당률 (%)",
                                                min_value=0,
                                                max_value=100,
                                                value=int(d_weight),
                                                key=f"edit_sub_prog_{task_id}_{i}",
                                            )
                                        if lbl.strip():
                                            edit_subtasks.append(
                                                (lbl.strip(), weight_val, d_done)
                                            )

                                    # 🔹 저장 / 취소
                                    b1, b2 = st.columns(2, gap="small")
                                    with b1:
                                        if st.button(
                                            "저장",
                                            key=f"save_edit_{task_id}",
                                            use_container_width=True,
                                        ):
                                            if edit_subtasks:
                                                new_desc = serialize_subtasks(
                                                    edit_subtasks
                                                )
                                                new_prog = (
                                                    calc_progress_from_subtasks(
                                                        edit_subtasks
                                                    )
                                                )
                                            else:
                                                new_desc = None
                                                new_prog = 0

                                            if new_prog == 0:
                                                new_status = "Todo"
                                            elif new_prog == 100:
                                                new_status = "Done"
                                            else:
                                                new_status = "In Progress"

                                            assignee_final = (
                                                None
                                                if assignee_val == "(없음)"
                                                else assignee_val
                                            )

                                            update_task(
                                                task_id,
                                                title=title_val.strip()
                                                or r["title"],
                                                status=new_status,
                                                description=new_desc,
                                                progress=int(new_prog),
                                                assignee=assignee_final,
                                                tags=tags_val.strip() or None,
                                                priority=priority_val,
                                                start_date=start_date_val.isoformat()
                                                if start_date_val
                                                else None,
                                                due_date=due_date_val.isoformat()
                                                if due_date_val
                                                else None,
                                            )
                                            st.session_state[edit_key] = False
                                            st.success("수정되었습니다.")
                                            st.rerun()
                                    with b2:
                                        if st.button(
                                            "취소",
                                            key=f"cancel_edit_{task_id}",
                                            use_container_width=True,
                                        ):
                                            st.session_state[edit_key] = False
                                            st.rerun()

            # 새 작업 추가
            st.divider()
            st.markdown("### ➕ 새 작업 추가")

            count_key = f"subtask_count_{part_id}"
            if count_key not in st.session_state:
                st.session_state[count_key] = 1

            with st.form(f"add_task_{part_id}"):

                # 🔹 제목 / 태그 / 우선순위 한 줄에 배치
                c_title, c_tag, c_pri = st.columns([2, 1, 1])
                with c_title:
                    title = st.text_input(
                        "제목*",
                        placeholder="예: API 연동 구현",
                        key=f"title_input_{part_id}",
                    )
                with c_tag:
                    tags = st.text_input(
                        "태그(쉼표 구분)",
                        placeholder="백엔드,UI 등",
                        key=f"tag_input_{part_id}",
                    )
                with c_pri:
                    new_priority = st.selectbox(
                        "우선순위",
                        priority_options,
                        index=1,  # 기본: Medium(보통)
                        format_func=lambda v: priority_labels.get(v, v),
                        key=f"priority_new_{part_id}",
                    )

                c1, c2 = st.columns(2)
                with c1:
                    assignee_choice = st.selectbox(
                        "담당자", user_options, key=f"assignee_{part_id}"
                    )
                with c2:
                    status = st.selectbox(
                        "상태",
                        ["Todo", "In Progress", "Done"],
                        key=f"status_new_{part_id}",
                    )

                c3, c4 = st.columns(2)
                with c3:
                    start_date = st.date_input(
                        "시작일",
                        value=selected_day,
                        key=f"start_{part_id}",
                    )
                with c4:
                    due_date = st.date_input(
                        "마감일",
                        value=selected_day,
                        key=f"due_{part_id}",
                    )

                sub_labels = []
                sub_weights = []
                for i in range(st.session_state[count_key]):
                    c_l, c_p = st.columns([3, 1])
                    with c_l:
                        lbl = st.text_input(
                            f"세부 작업 {i+1}",
                            key=f"new_sub_label_{part_id}_{i}",
                        )
                    with c_p:
                        prog_val = st.number_input(
                            "할당률 (%)",
                            min_value=0,
                            max_value=100,
                            value=0,
                            key=f"new_sub_prog_{part_id}_{i}",
                        )
                    if lbl.strip():
                        sub_labels.append(lbl.strip())
                        sub_weights.append(prog_val)

                b1, b2 = st.columns(2, gap="small")
                add_clicked = b1.form_submit_button(
                    "세부 작업 추가", use_container_width=True
                )
                save_clicked = b2.form_submit_button(
                    "저장", use_container_width=True
                )

                if add_clicked:
                    st.session_state[count_key] += 1

                if save_clicked:
                    if not title.strip():
                        st.error("제목은 필수입니다.")
                    else:
                        if assignee_choice == "(없음)":
                            assignee_val = None
                        else:
                            assignee_val = assignee_choice

                        subtasks_new = []
                        for lbl, w in zip(sub_labels, sub_weights):
                            done_flag = True if status == "Done" else False
                            subtasks_new.append((lbl, w, done_flag))

                        if subtasks_new:
                            description_str = serialize_subtasks(subtasks_new)
                        else:
                            description_str = None

                        if status == "Done":
                            progress = 100
                        else:
                            progress = 0

                        insert_task(
                            project_id=selected_project_id,
                            part_id=part_id,
                            title=title.strip(),
                            description=description_str,
                            assignee=assignee_val,
                            priority=new_priority,  # 🔹 선택한 우선순위 저장
                            status=status,
                            start_date=start_date.isoformat()
                            if start_date
                            else None,
                            due_date=due_date.isoformat()
                            if due_date
                            else None,
                            progress=int(progress),
                            tags=tags.strip() or None,
                        )
                        st.success("작업이 추가되었습니다.")

                        st.rerun()
