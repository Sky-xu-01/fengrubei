#include "udf.h"

#define HEAT_SOURCE_STRENGTH 600000.0
#define HEAT_SOURCE_TIME_THRESHOLD 1500.0

#define AC_T_LOW 297.15
#define AC_T_MID 299.15
#define AC_T_HIGH 302.15
#define AC_OUTLET_TEMP 293.15

#define AC_VEL_LOW 1.5
#define AC_VEL_MID 3.0
#define AC_VEL_HIGH 5.0

#define FAN_T_THRESHOLD 299.15
#define FAN_VEL_LOW 3.0
#define FAN_VEL_HIGH 5.0


#define SOLID_T_THRESHOLD 800.0


real get_solid_average_temp()
{
    Thread *t;
    cell_t c;
    real sum_temp = 0.0;
    real sum_vol = 0.0;
    real avg_temp = 0.0;
    Domain *domain = Get_Domain(1);

    thread_loop_c(t, domain)
    {
        if (SOLID_THREAD_P(t)) 
        {
            begin_c_loop(c, t)
            {
                sum_temp += C_T(c, t) * C_VOLUME(c, t);
                sum_vol += C_VOLUME(c, t);
            }
            end_c_loop(c, t)
        }
    }

    if (sum_vol > 0.0)
    {
        avg_temp = sum_temp / sum_vol;
    }
    else
    {
        avg_temp = 0.0;
        Message("Warning: Solid volume is zero!\n");
    }

    return avg_temp;
}

real get_domain_average_temp()
{
    Thread *t;
    cell_t c;
    real sum_temp = 0.0;
    real sum_vol = 0.0;
    real avg_temp;
    Domain *domain;
    domain = Get_Domain(1);

    thread_loop_c(t, domain)
    {
        if (FLUID_THREAD_P(t))
        {
            begin_c_loop(c, t)
            {
                sum_temp += C_T(c, t) * C_VOLUME(c, t);
                sum_vol += C_VOLUME(c, t);
            }
            end_c_loop(c, t)
        }
    }

    if (sum_vol > 0.0)
    {
        avg_temp = sum_temp / sum_vol;
    }
    else
    {
        avg_temp = AC_T_LOW;
        Message("Warning: Fluid volume is zero!\n");
    }

    return avg_temp;
}

DEFINE_SOURCE(heat_source_energy, c, t, dS, eqn)
{
    real source;
    real current_time = CURRENT_TIME;
    
    real solid_avg_temp = get_solid_average_temp();

  
    if (current_time < HEAT_SOURCE_TIME_THRESHOLD && solid_avg_temp <= SOLID_T_THRESHOLD)
    {
        source = HEAT_SOURCE_STRENGTH;
    }
    else
    {
        source = 0.0;
    }

    dS[eqn] = 0.0;

    if (fmod(current_time, 100.0) < 1e-6)
    {
        Message("Heat source status: %s (Time: %.1fs, Solid Temp: %.2f K, Strength: %.1e W/m³)\n",
                (current_time < HEAT_SOURCE_TIME_THRESHOLD && solid_avg_temp <= SOLID_T_THRESHOLD) ? "ON" : "OFF",
                current_time, solid_avg_temp, source);
    }

    return source;
}

DEFINE_PROFILE(aircon_velocity, thread, index)
{
    face_t f;
    real avg_temp = get_domain_average_temp();
    real vel = 0.0;

    if (avg_temp >= AC_T_LOW && avg_temp < AC_T_MID)
    {
        vel = AC_VEL_LOW;
    }
    else if (avg_temp >= AC_T_MID && avg_temp < AC_T_HIGH)
    {
        vel = AC_VEL_MID;
    }
    else if (avg_temp >= AC_T_HIGH)
    {
        vel = AC_VEL_HIGH;
    }

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, index) = vel;
    }
    end_f_loop(f, thread)

    Message("AC: Avg temp=%.2f K, Velocity=%.2f m/s\n", avg_temp, vel);
}

DEFINE_PROFILE(aircon_temperature, thread, index)
{
    face_t f;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, index) = AC_OUTLET_TEMP;
    }
    end_f_loop(f, thread)

    Message("AC outlet temperature: %.2f K\n", AC_OUTLET_TEMP);
}

DEFINE_PROFILE(fan_velocity, thread, index)
{
    face_t f;
    real avg_temp = get_domain_average_temp();
    real fan_vel;

    if (avg_temp <= FAN_T_THRESHOLD)
    {
        fan_vel = FAN_VEL_LOW;
    }
    else
    {
        fan_vel = FAN_VEL_HIGH;
    }

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, index) = fan_vel;
    }
    end_f_loop(f, thread)

    Message("Fan: Avg temp=%.2f K, Velocity=%.2f m/s\n", avg_temp, fan_vel);
}

DEFINE_PROFILE(fan_temperature, thread, index)
{
    face_t f;
    real avg_temp = get_domain_average_temp();

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, index) = avg_temp;
    }
    end_f_loop(f, thread)

    Message("Fan outlet temperature: %.2f K\n", avg_temp);
}