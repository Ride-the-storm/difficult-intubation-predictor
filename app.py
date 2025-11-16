from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_login import LoginManager, current_user, login_required
from flask_wtf.csrf import CSRFProtect, generate_csrf   # ✅ AGGIUNGI QUESTO
from database import db, User, IntubationData
from model import DifficultIntubationModel
from auth import auth_bp

app = Flask(__name__)
csrf = CSRFProtect(app)  # attiva il CSRF su tutte le view

# ✅ ABILITA CSRF
csrf = CSRFProtect(app)

# ✅ CONTEXT PROCESSOR PER I TEMPLATE
@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf)
# Database configuration
database_url = os.environ.get('DATABASE_URL')

if database_url:
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    if "postgresql" in database_url:
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "connect_args": {"sslmode": os.environ.get("PGSSLMODE", "prefer")}
        }
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# ---- Security & session hardening ----
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "1") == "1",
    REMEMBER_COOKIE_HTTPONLY=True,
)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'  # CAMBIA IN 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize model
model = DifficultIntubationModel(retrain_threshold=5)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Register blueprints
app.register_blueprint(auth_bp)  # AGGIUNGI QUESTA RIGA

# Le tue route esistenti rimangono uguali...

from flask import jsonify
from sqlalchemy import func, inspect

def _safe_numeric_columns(model):
    insp = inspect(model)
    cols = []
    for c in insp.columns:
        try:
            if hasattr(c.type, 'python_type'):
                pt = c.type.python_type
                if pt in (int, float):
                    cols.append(c.name)
        except NotImplementedError:
            continue
    return cols

@app.route('/api/dashboard-stats')
@login_required
def api_dashboard_stats():
    # Basic counts
    total = IntubationData.query.filter_by(user_id=current_user.id).count()

    # Attempt to compute label prevalence and metrics if columns exist
    # Expected optional columns: 'label' (0/1), 'predicted' (0/1), 'created_at' (datetime)
    # We'll compute sensitivity over time by month if data present.
    # Fallbacks are empty series.
    from collections import defaultdict
    label_col = None
    pred_col = None
    created_col = None

    # Probe model columns via SQLAlchemy
    insp = inspect(IntubationData)
    colnames = [c.name for c in insp.columns]
    for name in colnames:
        low = name.lower()
        if low in ('label', 'outcome', 'is_difficult', 'difficult'):
            label_col = name
        if low in ('predicted', 'prediction', 'y_pred'):
            pred_col = name
        if low in ('created_at', 'timestamp', 'created'):
            created_col = name

    def _q_all(cols):
        # returns list of dicts with requested columns plus id
        qcols = [getattr(IntubationData, c) for c in cols if hasattr(IntubationData, c)]
        if not qcols:
            return []
        rows = IntubationData.query.with_entities(*qcols).filter_by(user_id=current_user.id).all()
        return [dict(zip(cols, r)) for r in rows]

    # Numeric summary
    numeric_cols = [c for c in _safe_numeric_columns(IntubationData) if c not in ('id', 'user_id')]
    numeric_summary = {}
    if numeric_cols:
        # Compute count, mean, std, min, max per column from DB using SQL
        for c in numeric_cols:
            col = getattr(IntubationData, c)
            agg = db.session.query(
                func.count(col),
                func.avg(col),
                func.stddev_pop(col),
                func.min(col),
                func.max(col)
            ).filter_by(user_id=current_user.id).one()
            numeric_summary[c] = {
                "count": int(agg[0] or 0),
                "mean": float(agg[1]) if agg[1] is not None else None,
                "std": float(agg[2]) if agg[2] is not None else None,
                "min": float(agg[3]) if agg[3] is not None else None,
                "max": float(agg[4]) if agg[4] is not None else None,
            }

    # Compute metrics if we have labels
    prevalence = None
    confusion = None
    sensitivity_progress = []  # list of {period, sensitivity}
    if label_col:
        # prevalence
        positives = db.session.query(func.sum(getattr(IntubationData, label_col))).filter_by(user_id=current_user.id).scalar()
        total_lab = db.session.query(func.count(getattr(IntubationData, label_col))).filter_by(user_id=current_user.id).scalar()
        if positives is not None and total_lab:
            prevalence = float(positives) / float(total_lab)

        # confusion if we also have predictions
        if pred_col:
            tp = db.session.query(func.sum(func.case([(func.and_(getattr(IntubationData, label_col)==1, getattr(IntubationData, pred_col)==1), 1)], else_=0))).filter_by(user_id=current_user.id).scalar() or 0
            tn = db.session.query(func.sum(func.case([(func.and_(getattr(IntubationData, label_col)==0, getattr(IntubationData, pred_col)==0), 1)], else_=0))).filter_by(user_id=current_user.id).scalar() or 0
            fp = db.session.query(func.sum(func.case([(func.and_(getattr(IntubationData, label_col)==0, getattr(IntubationData, pred_col)==1), 1)], else_=0))).filter_by(user_id=current_user.id).scalar() or 0
            fn = db.session.query(func.sum(func.case([(func.and_(getattr(IntubationData, label_col)==1, getattr(IntubationData, pred_col)==0), 1)], else_=0))).filter_by(user_id=current_user.id).scalar() or 0
            confusion = {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

            # sensitivity over time if we have created_at
            if created_col:
                # group by YYYY-MM
                period = func.to_char(getattr(IntubationData, created_col), 'YYYY-MM')
                groups = db.session.query(
                    period.label('period'),
                    func.sum(func.case([(func.and_(getattr(IntubationData, label_col)==1, getattr(IntubationData, pred_col)==1), 1)], else_=0)).label('tp'),
                    func.sum(func.case([(getattr(IntubationData, label_col)==1, 1)], else_=0)).label('pos')
                ).filter_by(user_id=current_user.id).group_by(period).order_by(period).all()
                for g in groups:
                    sens = float(g.tp)/float(g.pos) if g.pos else None
                    sensitivity_progress.append({"period": g.period, "sensitivity": sens})

    # Correlation curve: compute correlation matrix among numeric columns
    correlation = []
    if len(numeric_cols) >= 2:
        # fallback: sample raw rows and compute pairwise Pearson via SQL aggregates would be heavy;
        # instead, we materialize minimal records into Python if count is small; otherwise skip.
        # We'll try to fetch up to 500 rows for these columns.
        rows = db.session.query(*[getattr(IntubationData, c) for c in numeric_cols]).filter_by(user_id=current_user.id).limit(500).all()
        if rows:
            import math
            import statistics as st
            data = list(map(lambda r: [float(x) if x is not None else None for x in r], rows))
            # simple pairwise pearson ignoring None
            def pearson(x, y):
                pairs = [(a,b) for a,b in zip(x,y) if a is not None and b is not None]
                if len(pairs) < 3:
                    return None
                xs, ys = zip(*pairs)
                mx, my = st.mean(xs), st.mean(ys)
                num = sum((a-mx)*(b-my) for a,b in pairs)
                denx = math.sqrt(sum((a-mx)**2 for a in xs))
                deny = math.sqrt(sum((b-my)**2 for b in ys))
                if denx == 0 or deny == 0:
                    return 0.0
                return num/(denx*deny)
            for i, ci in enumerate(numeric_cols):
                for j, cj in enumerate(numeric_cols):
                    corr = pearson([row[i] for row in data], [row[j] for row in data])
                    correlation.append({"x": ci, "y": cj, "value": corr})
    return jsonify({
        "count": total,
        "prevalence": prevalence,
        "confusion": confusion,
        "numeric_summary": numeric_summary,
        "sensitivity_progress": sensitivity_progress,
        "correlation": correlation,
        "numeric_columns": numeric_cols,
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    model_info = model.get_model_info()
    user_data_count = IntubationData.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html', 
                         model_info=model_info, 
                         user_data_count=user_data_count)

# ... tutte le altre route rimangono uguali


@click.command("create-admin")
@with_appcontext
def create_admin():
    """Create an admin user using env ADMIN_USERNAME/ADMIN_PASSWORD/ADMIN_EMAIL."""
    username = os.environ.get("ADMIN_USERNAME")
    password = os.environ.get("ADMIN_PASSWORD")
    email = os.environ.get("ADMIN_EMAIL", "admin@example.com")
    if not (username and password):
        raise click.UsageError("Set ADMIN_USERNAME and ADMIN_PASSWORD")
    if not User.query.filter_by(username=username).first():
        admin = User(username=username, email=email, is_admin=True, is_medical_professional=True)
        admin.set_password(password)
        db.session.add(admin)
        db.session.commit()
        click.echo(f"Admin created: {username}")
    else:
        click.echo("Admin already exists.")
app.cli.add_command(create_admin)

@click.command("init-model")
@with_appcontext
def init_model():
    """Load and, if needed, train the ML model."""
    model.load_model()
    if not model.is_trained:
        click.echo("Training initial model...")
        model.train(force_retrain=True)
    click.echo("Model initialized.")
app.cli.add_command(init_model)

@click.command("init-db")
@with_appcontext
def init_db_cli():
    """Create all database tables."""
    db.create_all()
    click.echo("Database initialized.")
app.cli.add_command(init_db_cli)


def init_db():
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                is_admin=True,
                is_medical_professional=True
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created: admin/admin123")
        
        # Initialize model
        model.load_model()
        if not model.is_trained:
            print("Training initial model...")
            model.train(force_retrain=True)

# Initialize via CLI or __main__

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
