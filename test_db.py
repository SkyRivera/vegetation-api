import asyncio
from sqlalchemy import text
from app.db.database import init_db


async def test_connection():
    try:
        print("Intentando conectar a la base de datos...")
        '''
        # 1. Probar conectividad ejecutando un query simple (SELECT 1)
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1;"))
            print("✅ Conexión exitosa a PostgreSQL!")
            print(f"Respuesta de la DB: {result.scalar()}")'''

        # 2. Inicializar tablas (opcional: crea la tabla 'predictions' si no existe)
        print("\nVerificando / creando tablas en la base de datos...")
        await init_db()
        print("✅ Tablas sincronizadas correctamente.")

    except Exception as e:
        print("\n❌ Error al conectar a la base de datos:")
        print(e)

if __name__ == "__main__":
    asyncio.run(test_connection())
